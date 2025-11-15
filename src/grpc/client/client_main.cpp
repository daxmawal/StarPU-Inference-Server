#include <grpcpp/grpcpp.h>
#include <torch/script.h>

#include <algorithm>
#include <chrono>
#include <format>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "client_args.hpp"
#include "inference_client.hpp"
#include "utils/input_generator.hpp"
#include "utils/logger.hpp"

namespace {
constexpr std::size_t kSummaryElementCount = 10;

using OutputSummary = std::vector<std::vector<double>>;

void
append_ivalue(const c10::IValue& value, std::vector<torch::Tensor>& outputs)
{
  if (value.isTensor()) {
    outputs.emplace_back(value.toTensor());
    return;
  }
  if (value.isTensorList()) {
    const auto tensor_list = value.toTensorList();
    outputs.insert(outputs.end(), tensor_list.begin(), tensor_list.end());
    return;
  }
  if (value.isTuple()) {
    for (const auto& element : value.toTuple()->elements()) {
      append_ivalue(element, outputs);
    }
    return;
  }
  if (value.isList()) {
    for (const auto& element : value.toList()) {
      append_ivalue(element, outputs);
    }
    return;
  }
  if (value.isGenericDict()) {
    for (const auto& item : value.toGenericDict()) {
      append_ivalue(item.value(), outputs);
    }
    return;
  }

  throw std::invalid_argument("Unsupported model output type");
}

auto
extract_tensors_from_output(const c10::IValue& value)
    -> std::vector<torch::Tensor>
{
  std::vector<torch::Tensor> outputs;
  append_ivalue(value, outputs);
  return outputs;
}

auto
summarize_tensor(const torch::Tensor& tensor, std::size_t max_values)
    -> std::vector<double>
{
  auto cpu_tensor = tensor.detach().cpu().contiguous();
  auto as_double = cpu_tensor.to(torch::kDouble);
  auto flattened = as_double.view({-1});
  const auto total = static_cast<std::size_t>(flattened.numel());
  const auto count = std::min(max_values, total);

  std::vector<double> summary;
  summary.reserve(count);
  const double* data_ptr = flattened.data_ptr<double>();
  const std::span<const double> tensor_values(data_ptr, total);
  for (const double value : tensor_values.first(count)) {
    summary.push_back(value);
  }
  return summary;
}

auto
summarize_outputs(const std::vector<torch::Tensor>& outputs) -> OutputSummary
{
  OutputSummary summary;
  summary.reserve(outputs.size());
  for (const auto& tensor : outputs) {
    summary.emplace_back(summarize_tensor(tensor, kSummaryElementCount));
  }
  return summary;
}

auto
run_client_reference_inference(
    torch::jit::script::Module& model,
    const std::vector<torch::Tensor>& inputs) -> OutputSummary
{
  c10::InferenceMode guard;
  std::vector<torch::IValue> ivals;
  ivals.reserve(inputs.size());
  for (const auto& tensor : inputs) {
    ivals.emplace_back(tensor);
  }
  const c10::IValue result = model.forward(ivals);
  auto tensors = extract_tensors_from_output(result);
  return summarize_outputs(tensors);
}
}  // namespace

auto
main(int argc, char* argv[]) -> int
{
  std::vector<const char*> const_argv(argv, argv + argc);
  std::span<const char*> args{const_argv};
  const starpu_server::ClientConfig config =
      starpu_server::parse_client_args(args);
  if (config.show_help) {
    starpu_server::display_client_help(args.front());
    return 0;
  }
  if (!config.valid) {
    starpu_server::log_error("Invalid program options.");
    return 1;
  }
  grpc::ChannelArguments ch_args;
  const int max_msg_size = 32 * 1024 * 1024;
  ch_args.SetMaxReceiveMessageSize(max_msg_size);
  ch_args.SetMaxSendMessageSize(max_msg_size);

  auto channel = grpc::CreateCustomChannel(
      config.server_address, grpc::InsecureChannelCredentials(), ch_args);

  starpu_server::InferenceClient client(channel, config.verbosity);

  if (!client.ServerIsLive()) {
    return 1;
  }

  if (!client.ServerIsReady()) {
    return 1;
  }

  if (!client.ModelIsReady({config.model_name, config.model_version})) {
    return 1;
  }

  std::unique_ptr<torch::jit::script::Module> reference_model;
  if (!config.client_model_path.empty()) {
    try {
      auto model = torch::jit::load(config.client_model_path);
      model.eval();
      starpu_server::log_info(
          config.verbosity, std::format(
                                "Loaded client model '{}' for local validation",
                                config.client_model_path));
      reference_model =
          std::make_unique<torch::jit::script::Module>(std::move(model));
    }
    catch (const c10::Error& e) {
      starpu_server::log_error(std::format(
          "Failed to load client model '{}': {}", config.client_model_path,
          e.what()));
    }
  }

  constexpr int NUM_TENSORS = 5;
  std::vector<std::vector<torch::Tensor>> tensor_pool;
  tensor_pool.reserve(NUM_TENSORS);
  std::vector<std::optional<OutputSummary>> reference_summaries;
  reference_summaries.reserve(NUM_TENSORS);
  auto validation_active = static_cast<bool>(reference_model);
  for (int i = 0; i < NUM_TENSORS; ++i) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(config.inputs.size());
    for (size_t j = 0; j < config.inputs.size(); ++j) {
      const auto& in_cfg = config.inputs[j];
      tensors.push_back(starpu_server::input_generator::generate_random_tensor(
          in_cfg.shape, in_cfg.type, j));
    }
    std::optional<OutputSummary> outputs_summary;
    if (validation_active && reference_model) {
      try {
        outputs_summary =
            run_client_reference_inference(*reference_model, tensors);
      }
      catch (const std::exception& e) {
        starpu_server::log_warning(std::format(
            "Disabling client-side validation after failure: {}", e.what()));
        validation_active = false;
      }
    }
    reference_summaries.emplace_back(std::move(outputs_summary));
    tensor_pool.push_back(std::move(tensors));
  }

  thread_local std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution dist(0, NUM_TENSORS - 1);

  std::jthread cq_thread(
      &starpu_server::InferenceClient::AsyncCompleteRpc, &client);

  auto next_time = std::chrono::steady_clock::now();
  const auto delay = std::chrono::microseconds(config.delay_us);
  for (int i = 0; i < config.request_nb; ++i) {
    std::this_thread::sleep_until(next_time);
    next_time += delay;

    const auto idx = static_cast<size_t>(dist(rng));
    const auto& expected_outputs = reference_summaries[idx];
    client.AsyncModelInfer(tensor_pool[idx], config, expected_outputs);
  }

  client.Shutdown();

  return 0;
}
