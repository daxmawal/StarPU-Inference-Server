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
using TensorPool = std::vector<std::vector<torch::Tensor>>;
using ReferenceSummaries = std::vector<std::optional<OutputSummary>>;

struct ScheduleSegment {
  std::chrono::microseconds delta{0};
  std::size_t repeat = 0;
  std::optional<std::size_t> input_id = std::nullopt;
};

struct PreparedClientInputs {
  TensorPool tensor_pool;
  ReferenceSummaries reference_summaries;
};

struct SchedulePlan {
  std::vector<ScheduleSegment> segments;
  std::size_t scheduled_requests = 0;
  std::size_t target_requests = 0;
};

constexpr std::size_t kTensorPoolSize = 5;
constexpr int kClientMaxMessageSize = 32 * 1024 * 1024;

auto
trim_ascii(const std::string_view value) -> std::string_view
{
  auto is_space = [](const unsigned char character) {
    return std::isspace(character) != 0;
  };

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

void
validate_schedule_header(const std::vector<std::string_view>& fields)
{
  if (fields.size() != 2 && fields.size() != 3) {
    throw std::invalid_argument(
        "header must be 'delta_us,repeat' or 'delta_us,repeat,input_id'");
  }
  if (fields[0] != "delta_us" || fields[1] != "repeat") {
    throw std::invalid_argument("header must start with 'delta_us,repeat'");
  }
  if (fields.size() == 3 && fields[2] != "input_id") {
    throw std::invalid_argument(
        "third header column must be 'input_id' when present");
  }
}

auto
parse_schedule_input_id(
    const std::vector<std::string_view>& fields, const bool has_input_id,
    const std::size_t num_inputs,
    const std::size_t line_no) -> std::optional<std::size_t>
{
  if (!has_input_id) {
    return std::nullopt;
  }

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
  return parsed_idx;
}

auto
parse_schedule_segment(
    const std::vector<std::string_view>& fields,
    const std::size_t expected_columns, const bool has_input_id,
    const std::size_t num_inputs, const std::size_t line_no) -> ScheduleSegment
{
  if (fields.size() != expected_columns) {
    throw std::invalid_argument(std::format(
        "line {}: expected {} columns but got {}", line_no, expected_columns,
        fields.size()));
  }

  const int64_t delta_us = parse_int_field(fields[0], "delta_us", line_no);
  if (delta_us < 0) {
    throw std::invalid_argument(
        std::format("line {}: 'delta_us' must be >= 0", line_no));
  }

  const int64_t repeat = parse_int_field(fields[1], "repeat", line_no);
  if (repeat <= 0) {
    throw std::invalid_argument(
        std::format("line {}: 'repeat' must be > 0", line_no));
  }

  return ScheduleSegment{
      std::chrono::microseconds(delta_us), static_cast<std::size_t>(repeat),
      parse_schedule_input_id(fields, has_input_id, num_inputs, line_no)};
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
      validate_schedule_header(fields);
      expected_columns = fields.size();
      has_input_id = (fields.size() == 3);
      header_seen = true;
      continue;
    }

    segments.push_back(parse_schedule_segment(
        fields, expected_columns, has_input_id, num_inputs, line_no));
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

auto
make_channel_arguments() -> grpc::ChannelArguments
{
  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(kClientMaxMessageSize);
  args.SetMaxSendMessageSize(kClientMaxMessageSize);
  return args;
}

auto
ensure_server_ready(
    starpu_server::InferenceClient& client,
    const starpu_server::ClientConfig& config) -> bool
{
  if (!client.ServerIsLive()) {
    return false;
  }
  if (!client.ServerIsReady()) {
    return false;
  }
  return client.ModelIsReady({config.model_name, config.model_version});
}

auto
load_reference_model(const starpu_server::ClientConfig& config)
    -> std::unique_ptr<torch::jit::script::Module>
{
  if (config.client_model_path.empty()) {
    return nullptr;
  }

  try {
    auto model = torch::jit::load(config.client_model_path);
    model.eval();
    starpu_server::log_info(
        config.verbosity, std::format(
                              "Loaded client model '{}' for local validation",
                              config.client_model_path));
    return std::make_unique<torch::jit::script::Module>(std::move(model));
  }
  catch (const c10::Error& error) {
    starpu_server::log_error(std::format(
        "Failed to load client model '{}': {}", config.client_model_path,
        error.what()));
    return nullptr;
  }
}

auto
generate_input_tensors(const starpu_server::ClientConfig& config)
    -> std::vector<torch::Tensor>
{
  std::vector<torch::Tensor> tensors;
  tensors.reserve(config.inputs.size());
  for (std::size_t input_idx = 0; input_idx < config.inputs.size();
       ++input_idx) {
    const auto& input_config = config.inputs[input_idx];
    tensors.push_back(starpu_server::input_generator::generate_random_tensor(
        input_config.shape, input_config.type, input_idx));
  }
  return tensors;
}

auto
maybe_build_reference_summary(
    torch::jit::script::Module* reference_model, bool& validation_active,
    const std::vector<torch::Tensor>& tensors) -> std::optional<OutputSummary>
{
  if (!validation_active || reference_model == nullptr) {
    return std::nullopt;
  }

  try {
    return run_client_reference_inference(*reference_model, tensors);
  }
  catch (const std::exception& error) {
    starpu_server::log_warning(std::format(
        "Disabling client-side validation after failure: {}", error.what()));
    validation_active = false;
    return std::nullopt;
  }
}

auto
prepare_client_inputs(
    const starpu_server::ClientConfig& config,
    torch::jit::script::Module* reference_model) -> PreparedClientInputs
{
  PreparedClientInputs prepared{};
  prepared.tensor_pool.reserve(kTensorPoolSize);
  prepared.reference_summaries.reserve(kTensorPoolSize);

  bool validation_active = (reference_model != nullptr);
  for (std::size_t tensor_idx = 0; tensor_idx < kTensorPoolSize; ++tensor_idx) {
    auto tensors = generate_input_tensors(config);
    prepared.reference_summaries.emplace_back(maybe_build_reference_summary(
        reference_model, validation_active, tensors));
    prepared.tensor_pool.push_back(std::move(tensors));
  }
  return prepared;
}

void
dispatch_request(
    starpu_server::InferenceClient& client,
    const starpu_server::ClientConfig& config, const TensorPool& tensor_pool,
    const ReferenceSummaries& reference_summaries, const std::size_t input_idx)
{
  client.AsyncModelInfer(
      tensor_pool[input_idx], config, reference_summaries[input_idx]);
}

auto
random_input_index(
    std::uniform_int_distribution<int>& distribution,
    std::mt19937& rng) -> std::size_t
{
  return static_cast<std::size_t>(distribution(rng));
}

void
dispatch_fixed_rate_requests(
    starpu_server::InferenceClient& client,
    const starpu_server::ClientConfig& config, const TensorPool& tensor_pool,
    const ReferenceSummaries& reference_summaries,
    std::uniform_int_distribution<int>& distribution, std::mt19937& rng)
{
  auto next_time = std::chrono::steady_clock::now();
  const auto delay = std::chrono::microseconds(config.delay_us);
  for (int request_idx = 0; request_idx < config.request_nb; ++request_idx) {
    std::this_thread::sleep_until(next_time);
    next_time += delay;

    dispatch_request(
        client, config, tensor_pool, reference_summaries,
        random_input_index(distribution, rng));
  }
}

auto
schedule_request_cap_suffix(
    const starpu_server::ClientConfig& config,
    const std::size_t target_requests) -> std::string
{
  if (!config.request_nb_explicit) {
    return {};
  }
  return std::format(", capped to {}", target_requests);
}

auto
try_load_schedule_plan(
    const starpu_server::ClientConfig& config,
    const std::size_t num_inputs) -> std::optional<SchedulePlan>
{
  SchedulePlan plan{};
  try {
    plan.segments = load_schedule_csv(config.schedule_csv_path, num_inputs);
  }
  catch (const std::exception& error) {
    starpu_server::log_error(std::format(
        "Failed to parse schedule CSV '{}': {}", config.schedule_csv_path,
        error.what()));
    return std::nullopt;
  }

  try {
    plan.scheduled_requests = count_scheduled_requests(plan.segments);
  }
  catch (const std::exception& error) {
    starpu_server::log_error(std::format(
        "Failed to compute scheduled request count for '{}': {}",
        config.schedule_csv_path, error.what()));
    return std::nullopt;
  }

  if (config.delay_us > 0) {
    starpu_server::log_info(
        config.verbosity,
        "--schedule-csv is set; --delay is ignored in schedule mode.");
  }

  plan.target_requests = plan.scheduled_requests;
  if (config.request_nb_explicit) {
    plan.target_requests = std::min(
        plan.target_requests, static_cast<std::size_t>(config.request_nb));
  }

  starpu_server::log_info(
      config.verbosity,
      std::format(
          "Loaded schedule CSV '{}' with {} segment(s), {} planned request(s)"
          "{}.",
          config.schedule_csv_path, plan.segments.size(),
          plan.scheduled_requests,
          schedule_request_cap_suffix(config, plan.target_requests)));
  return plan;
}

auto
scheduled_input_index(
    const ScheduleSegment& segment,
    std::uniform_int_distribution<int>& distribution,
    std::mt19937& rng) -> std::size_t
{
  if (!segment.input_id.has_value()) {
    return random_input_index(distribution, rng);
  }
  return segment.input_id.value();
}

void
dispatch_scheduled_requests(
    starpu_server::InferenceClient& client,
    const starpu_server::ClientConfig& config, const TensorPool& tensor_pool,
    const ReferenceSummaries& reference_summaries, const SchedulePlan& plan,
    std::uniform_int_distribution<int>& distribution, std::mt19937& rng)
{
  auto next_time = std::chrono::steady_clock::now();
  std::size_t sent_requests = 0;
  for (const auto& segment : plan.segments) {
    for (std::size_t repeat_idx = 0;
         repeat_idx < segment.repeat && sent_requests < plan.target_requests;
         ++repeat_idx) {
      std::this_thread::sleep_until(next_time);
      next_time += segment.delta;

      dispatch_request(
          client, config, tensor_pool, reference_summaries,
          scheduled_input_index(segment, distribution, rng));
      ++sent_requests;
    }
    if (sent_requests >= plan.target_requests) {
      return;
    }
  }
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

  auto channel = grpc::CreateCustomChannel(
      config.server_address, grpc::InsecureChannelCredentials(),
      make_channel_arguments());

  starpu_server::InferenceClient client(channel, config.verbosity);

  if (!ensure_server_ready(client, config)) {
    return 1;
  }

  auto reference_model = load_reference_model(config);
  const auto prepared_inputs =
      prepare_client_inputs(config, reference_model.get());

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> input_distribution(
      0, static_cast<int>(prepared_inputs.tensor_pool.size() - 1));

  std::jthread cq_thread(
      &starpu_server::InferenceClient::AsyncCompleteRpc, &client);

  if (config.schedule_csv_path.empty()) {
    dispatch_fixed_rate_requests(
        client, config, prepared_inputs.tensor_pool,
        prepared_inputs.reference_summaries, input_distribution, rng);
  } else {
    const auto schedule_plan =
        try_load_schedule_plan(config, prepared_inputs.tensor_pool.size());
    if (!schedule_plan.has_value()) {
      client.Shutdown();
      return 1;
    }

    dispatch_scheduled_requests(
        client, config, prepared_inputs.tensor_pool,
        prepared_inputs.reference_summaries, *schedule_plan, input_distribution,
        rng);
  }

  client.Shutdown();
  if (cq_thread.joinable()) {
    cq_thread.join();
  }

  if (!config.summary_json_path.empty() &&
      !client.write_summary_json(config.summary_json_path)) {
    return 1;
  }

  return 0;
}
