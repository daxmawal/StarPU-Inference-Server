#include <grpcpp/grpcpp.h>
#include <torch/script.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "client_args.hpp"
#include "inference_client.hpp"
#include "utils/input_generator.hpp"
#include "utils/logger.hpp"

inline namespace client_main_detail {
using OutputSummary = std::vector<std::vector<double>>;

struct ScheduleSegment {
  std::chrono::microseconds delta{0};
  std::size_t repeat = 0;
  std::optional<std::size_t> input_id = std::nullopt;
};

auto
trim_ascii(const std::string_view value) -> std::string_view
{
  auto is_space = [](const unsigned char c) { return std::isspace(c) != 0; };

  std::size_t begin = 0;
  while (begin < value.size() &&
         is_space(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }

  std::size_t end = value.size();
  while (end > begin && is_space(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(begin, end - begin);
}

auto
split_csv_fields(const std::string& line) -> std::vector<std::string_view>
{
  std::vector<std::string_view> fields;
  const std::string_view view(line);
  std::size_t begin = 0;
  while (true) {
    const auto comma = view.find(',', begin);
    if (comma == std::string_view::npos) {
      fields.push_back(trim_ascii(view.substr(begin)));
      break;
    }
    fields.push_back(trim_ascii(view.substr(begin, comma - begin)));
    begin = comma + 1;
    if (begin == view.size()) {
      fields.emplace_back();
      break;
    }
  }
  return fields;
}

auto
parse_int_field(
    const std::string_view value, const char* field,
    const std::size_t line_no) -> int64_t
{
  if (value.empty()) {
    throw std::invalid_argument(
        std::format("line {}: '{}' cannot be empty", line_no, field));
  }

  std::size_t consumed = 0;
  const std::string owned(value);
  int64_t parsed = 0;
  try {
    parsed = std::stoll(owned, &consumed);
  }
  catch (const std::invalid_argument&) {
    throw std::invalid_argument(
        std::format("line {}: '{}' is not an integer", line_no, field));
  }
  catch (const std::out_of_range&) {
    throw std::out_of_range(
        std::format("line {}: '{}' is out of range", line_no, field));
  }
  if (consumed != owned.size()) {
    throw std::invalid_argument(
        std::format("line {}: '{}' must be a plain integer", line_no, field));
  }
  return parsed;
}

auto
load_schedule_csv(const std::string& path, const std::size_t num_inputs)
    -> std::vector<ScheduleSegment>
{
  std::ifstream stream(path);
  if (!stream.is_open()) {
    throw std::invalid_argument("cannot open file");
  }

  std::vector<ScheduleSegment> segments;
  std::string line;
  std::size_t line_no = 0;
  bool header_seen = false;
  std::size_t expected_columns = 0;
  bool has_input_id = false;

  while (std::getline(stream, line)) {
    ++line_no;
    const std::string_view stripped = trim_ascii(line);
    if (stripped.empty() || stripped.starts_with('#')) {
      continue;
    }

    const auto fields = split_csv_fields(line);
    if (!header_seen) {
      if (fields.size() != 2 && fields.size() != 3) {
        throw std::invalid_argument(
            "header must be 'delta_us,repeat' or "
            "'delta_us,repeat,input_id'");
      }
      if (fields[0] != "delta_us" || fields[1] != "repeat") {
        throw std::invalid_argument("header must start with 'delta_us,repeat'");
      }
      if (fields.size() == 3 && fields[2] != "input_id") {
        throw std::invalid_argument(
            "third header column must be 'input_id' when present");
      }
      expected_columns = fields.size();
      has_input_id = (fields.size() == 3);
      header_seen = true;
      continue;
    }

    if (fields.size() != expected_columns) {
      throw std::invalid_argument(std::format(
          "line {}: expected {} columns but got {}", line_no, expected_columns,
          fields.size()));
    }

    const int64_t delta_us = parse_int_field(fields[0], "delta_us", line_no);
    const int64_t repeat = parse_int_field(fields[1], "repeat", line_no);
    if (delta_us < 0) {
      throw std::invalid_argument(
          std::format("line {}: 'delta_us' must be >= 0", line_no));
    }
    if (repeat <= 0) {
      throw std::invalid_argument(
          std::format("line {}: 'repeat' must be > 0", line_no));
    }

    std::optional<std::size_t> input_id = std::nullopt;
    if (has_input_id) {
      const int64_t parsed_input_id =
          parse_int_field(fields[2], "input_id", line_no);
      if (parsed_input_id < 0) {
        throw std::invalid_argument(
            std::format("line {}: 'input_id' must be >= 0", line_no));
      }
      const auto parsed_idx = static_cast<std::size_t>(parsed_input_id);
      if (parsed_idx >= num_inputs) {
        throw std::invalid_argument(std::format(
            "line {}: 'input_id'={} out of range [0, {})", line_no, parsed_idx,
            num_inputs));
      }
      input_id = parsed_idx;
    }

    segments.push_back(ScheduleSegment{
        std::chrono::microseconds(delta_us), static_cast<std::size_t>(repeat),
        input_id});
  }

  if (!header_seen) {
    throw std::invalid_argument("CSV is empty or missing header");
  }
  if (segments.empty()) {
    throw std::invalid_argument("CSV has no schedule rows");
  }
  return segments;
}

auto
count_scheduled_requests(const std::vector<ScheduleSegment>& segments)
    -> std::size_t
{
  std::size_t total = 0;
  for (const auto& segment : segments) {
    if (total > std::numeric_limits<std::size_t>::max() - segment.repeat) {
      throw std::overflow_error("total request count overflow");
    }
    total += segment.repeat;
  }
  return total;
}

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
summarize_tensor(const torch::Tensor& tensor) -> std::vector<double>
{
  auto cpu_tensor = tensor.detach().cpu().contiguous();
  auto as_double = cpu_tensor.to(torch::kDouble);
  auto flattened = as_double.view({-1});
  const auto total = static_cast<std::size_t>(flattened.numel());

  std::vector<double> summary;
  summary.reserve(total);
  const double* data_ptr = flattened.data_ptr<double>();
  const std::span<const double> tensor_values(data_ptr, total);
  for (const double value : tensor_values) {
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
    summary.emplace_back(summarize_tensor(tensor));
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
}  // namespace client_main_detail

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

  if (config.schedule_csv_path.empty()) {
    auto next_time = std::chrono::steady_clock::now();
    const auto delay = std::chrono::microseconds(config.delay_us);
    for (int i = 0; i < config.request_nb; ++i) {
      std::this_thread::sleep_until(next_time);
      next_time += delay;

      const auto idx = static_cast<size_t>(dist(rng));
      const auto& expected_outputs = reference_summaries[idx];
      client.AsyncModelInfer(tensor_pool[idx], config, expected_outputs);
    }
  } else {
    std::vector<ScheduleSegment> segments;
    try {
      segments =
          load_schedule_csv(config.schedule_csv_path, tensor_pool.size());
    }
    catch (const std::exception& e) {
      starpu_server::log_error(std::format(
          "Failed to parse schedule CSV '{}': {}", config.schedule_csv_path,
          e.what()));
      client.Shutdown();
      return 1;
    }

    std::size_t scheduled_requests = 0;
    try {
      scheduled_requests = count_scheduled_requests(segments);
    }
    catch (const std::exception& e) {
      starpu_server::log_error(std::format(
          "Failed to compute scheduled request count for '{}': {}",
          config.schedule_csv_path, e.what()));
      client.Shutdown();
      return 1;
    }

    if (config.delay_us > 0) {
      starpu_server::log_info(
          config.verbosity,
          "--schedule-csv is set; --delay is ignored in schedule mode.");
    }

    std::size_t target_requests = scheduled_requests;
    if (config.request_nb_explicit) {
      target_requests =
          std::min<std::size_t>(target_requests, config.request_nb);
    }

    starpu_server::log_info(
        config.verbosity,
        std::format(
            "Loaded schedule CSV '{}' with {} segment(s), {} planned request(s)"
            "{}.",
            config.schedule_csv_path, segments.size(), scheduled_requests,
            config.request_nb_explicit
                ? std::format(", capped to {}", target_requests)
                : ""));

    auto next_time = std::chrono::steady_clock::now();
    std::size_t sent_requests = 0;
    for (const auto& segment : segments) {
      for (std::size_t i = 0;
           i < segment.repeat && sent_requests < target_requests; ++i) {
        std::this_thread::sleep_until(next_time);
        next_time += segment.delta;

        const auto input_idx = segment.input_id.has_value()
                                   ? *segment.input_id
                                   : static_cast<std::size_t>(dist(rng));
        const auto& expected_outputs = reference_summaries[input_idx];
        client.AsyncModelInfer(
            tensor_pool[input_idx], config, expected_outputs);
        ++sent_requests;
      }
      if (sent_requests >= target_requests) {
        break;
      }
    }
  }

  client.Shutdown();

  return 0;
}
