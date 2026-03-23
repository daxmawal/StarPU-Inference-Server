#include "inference_client.hpp"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "core/latency_statistics.hpp"
#include "utils/datatype_utils.hpp"
#include "utils/logger.hpp"
#include "utils/time_utils.hpp"

namespace starpu_server {
struct AsyncClientCall {
  inference::ModelInferResponse reply;
  grpc::ClientContext context;
  grpc::Status status;
  std::unique_ptr<
      grpc::ClientAsyncResponseReader<inference::ModelInferResponse>>
      response_reader = nullptr;
  std::chrono::system_clock::time_point start_time;
  int request_id = 0;
  std::size_t inference_count = 1;
  std::optional<InferenceClient::OutputSummary> expected_outputs;
};

inline namespace inference_client_detail {
constexpr double kOutputComparisonTolerance = 1e-4;
constexpr auto kJsonNumberPrecision = std::streamsize{10};

template <typename T>
void
append_converted_values(
    std::vector<double>& destination, std::string_view raw, std::size_t count)
{
  for (std::size_t idx = 0; idx < count; ++idx) {
    T value{};
    std::memcpy(&value, raw.data() + idx * sizeof(T), sizeof(T));
    destination.push_back(static_cast<double>(value));
  }
}

template <>
void
append_converted_values<c10::Half>(
    std::vector<double>& destination, std::string_view raw, std::size_t count)
{
  for (std::size_t idx = 0; idx < count; ++idx) {
    c10::Half value{};
    std::memcpy(
        &value, raw.data() + idx * sizeof(c10::Half), sizeof(c10::Half));
    destination.push_back(static_cast<double>(static_cast<float>(value)));
  }
}

template <>
void
append_converted_values<c10::BFloat16>(
    std::vector<double>& destination, std::string_view raw, std::size_t count)
{
  for (std::size_t idx = 0; idx < count; ++idx) {
    c10::BFloat16 value{};
    std::memcpy(
        &value, raw.data() + idx * sizeof(c10::BFloat16),
        sizeof(c10::BFloat16));
    destination.push_back(static_cast<double>(static_cast<float>(value)));
  }
}

auto
decode_output_values(
    const inference::ModelInferResponse::InferOutputTensor& tensor,
    std::string_view raw, std::size_t limit) -> std::vector<double>
{
  if (limit == 0) {
    return {};
  }

  const at::ScalarType type = datatype_to_scalar_type(tensor.datatype());
  const std::size_t elem_size = element_size(type);
  if (elem_size == 0) {
    throw std::invalid_argument("Unsupported element size");
  }

  const std::size_t available = raw.size() / elem_size;
  const std::size_t count = std::min(limit, available);

  std::vector<double> decoded;
  decoded.reserve(count);

  switch (type) {
    case at::kFloat:
      append_converted_values<float>(decoded, raw, count);
      break;
    case at::kDouble:
      append_converted_values<double>(decoded, raw, count);
      break;
    case at::kHalf:
      append_converted_values<c10::Half>(decoded, raw, count);
      break;
    case at::kBFloat16:
      append_converted_values<c10::BFloat16>(decoded, raw, count);
      break;
    case at::kInt:
      append_converted_values<int32_t>(decoded, raw, count);
      break;
    case at::kLong:
      append_converted_values<int64_t>(decoded, raw, count);
      break;
    case at::kShort:
      append_converted_values<int16_t>(decoded, raw, count);
      break;
    case at::kChar:
      append_converted_values<int8_t>(decoded, raw, count);
      break;
    case at::kByte:
    case at::kBool:
      append_converted_values<uint8_t>(decoded, raw, count);
      break;
    default:
      throw std::invalid_argument(
          std::format("Unsupported output datatype '{}'", tensor.datatype()));
  }

  return decoded;
}

auto
json_number(double value) -> std::string
{
  std::ostringstream stream;
  stream << std::setprecision(kJsonNumberPrecision) << value;
  return stream.str();
}

void
append_latency_summary_json(
    std::string& json, std::string_view key,
    const std::optional<InferenceClient::LatencySummary>& summary,
    bool& first_field)
{
  if (!first_field) {
    json += ",\n";
  }
  first_field = false;
  json += std::format("  \"{}\": ", key);

  if (!summary.has_value()) {
    json += "null";
    return;
  }

  json += std::format(
      R"({{"mean_ms": {}, "p50_ms": {}, "p85_ms": {}, "p95_ms": {}, "p100_ms": {}}})",
      json_number(summary->mean_ms), json_number(summary->p50_ms),
      json_number(summary->p85_ms), json_number(summary->p95_ms),
      json_number(summary->p100_ms));
}
}  // namespace inference_client_detail

InferenceClient::InferenceClient(
    std::shared_ptr<grpc::Channel>& channel, VerbosityLevel verbosity)
    : stub_(inference::GRPCInferenceService::NewStub(channel)),
      verbosity_(verbosity)
{
}

void
InferenceClient::record_latency(const LatencySample& sample)
{
  latency_records_.roundtrip_ms.push_back(sample.roundtrip_ms);
  latency_records_.server_overall_ms.push_back(sample.server_overall_ms);
  latency_records_.server_preprocess_ms.push_back(sample.server_preprocess_ms);
  latency_records_.server_queue_ms.push_back(sample.server_queue_ms);
  latency_records_.server_batch_ms.push_back(sample.server_batch_ms);
  latency_records_.server_submit_ms.push_back(sample.server_submit_ms);
  latency_records_.server_scheduling_ms.push_back(sample.server_scheduling_ms);
  latency_records_.server_codelet_ms.push_back(sample.server_codelet_ms);
  latency_records_.server_inference_ms.push_back(sample.server_inference_ms);
  latency_records_.server_callback_ms.push_back(sample.server_callback_ms);
  latency_records_.server_postprocess_ms.push_back(
      sample.server_postprocess_ms);
  latency_records_.server_job_total_ms.push_back(sample.server_job_total_ms);
  latency_records_.request_latency_ms.push_back(sample.request_latency_ms);
  latency_records_.response_latency_ms.push_back(sample.response_latency_ms);
  latency_records_.client_overhead_ms.push_back(sample.client_overhead_ms);
}

auto
InferenceClient::summarize_latencies(const std::vector<double>& values)
    -> std::optional<LatencySummary>
{
  const auto stats = compute_latency_statistics(values);
  if (!stats.has_value()) {
    return std::nullopt;
  }

  return LatencySummary{
      .mean_ms = stats->mean,
      .p50_ms = stats->p50,
      .p85_ms = stats->p85,
      .p95_ms = stats->p95,
      .p100_ms = stats->p100,
  };
}

auto
InferenceClient::summary() const -> Summary
{
  Summary result{};
  result.requests_sent = total_requests_sent_;
  result.requests_ok = success_requests_;
  result.requests_rejected = rejected_requests_;
  result.requests_handled = success_requests_ + rejected_requests_;
  result.inference_count = total_inference_count_;
  result.response_count = latency_records_.roundtrip_ms.size();
  result.roundtrip_latency = summarize_latencies(latency_records_.roundtrip_ms);
  result.server_latency.overall =
      summarize_latencies(latency_records_.server_overall_ms);
  result.server_latency.preprocess =
      summarize_latencies(latency_records_.server_preprocess_ms);
  result.server_latency.queue =
      summarize_latencies(latency_records_.server_queue_ms);
  result.server_latency.batching =
      summarize_latencies(latency_records_.server_batch_ms);
  result.server_latency.submit =
      summarize_latencies(latency_records_.server_submit_ms);
  result.server_latency.scheduling =
      summarize_latencies(latency_records_.server_scheduling_ms);
  result.server_latency.codelet =
      summarize_latencies(latency_records_.server_codelet_ms);
  result.server_latency.inference =
      summarize_latencies(latency_records_.server_inference_ms);
  result.server_latency.callback =
      summarize_latencies(latency_records_.server_callback_ms);
  result.server_latency.postprocess =
      summarize_latencies(latency_records_.server_postprocess_ms);
  result.server_latency.job_total =
      summarize_latencies(latency_records_.server_job_total_ms);
  result.request_latency =
      summarize_latencies(latency_records_.request_latency_ms);
  result.response_latency =
      summarize_latencies(latency_records_.response_latency_ms);
  result.client_overhead_latency =
      summarize_latencies(latency_records_.client_overhead_ms);

  if (first_request_time_.has_value() && last_response_time_.has_value() &&
      total_inference_count_ > 0) {
    const double elapsed_seconds =
        std::chrono::duration<double>(
            *last_response_time_ - *first_request_time_)
            .count();
    if (elapsed_seconds > 0.0) {
      result.elapsed_seconds = elapsed_seconds;
      result.throughput_rps =
          static_cast<double>(total_inference_count_) / elapsed_seconds;
    }
  }

  return result;
}

auto
InferenceClient::write_summary_json(const std::filesystem::path& path) const
    -> bool
{
  std::error_code status_ec;
  if (const auto parent = path.parent_path(); !parent.empty()) {
    std::filesystem::create_directories(parent, status_ec);
    if (status_ec) {
      log_error(std::format(
          "Failed to create summary directory '{}': {}", parent.string(),
          status_ec.message()));
      return false;
    }
  }

  std::ofstream stream(path);
  if (!stream.is_open()) {
    log_error(
        std::format("Failed to open summary JSON file '{}'", path.string()));
    return false;
  }

  const auto report = summary();
  std::string json;
  json += "{\n";
  json += std::format(
      "  \"requests\": {{\"sent\": {}, \"handled\": {}, \"ok\": {}, "
      "\"rejected\": {}}},\n",
      report.requests_sent, report.requests_handled, report.requests_ok,
      report.requests_rejected);
  json += std::format(
      "  \"response_count\": {},\n"
      "  \"inference_count\": {},\n",
      report.response_count, report.inference_count);
  json += "  \"elapsed_seconds\": ";
  json += report.elapsed_seconds.has_value()
              ? json_number(*report.elapsed_seconds)
              : "null";
  json += ",\n";
  json += "  \"throughput_rps\": ";
  json += report.throughput_rps.has_value()
              ? json_number(*report.throughput_rps)
              : "null";
  json += ",\n";
  json += "  \"latency_ms\": {\n";

  bool first_latency_field = true;
  append_latency_summary_json(
      json, "roundtrip", report.roundtrip_latency, first_latency_field);
  append_latency_summary_json(
      json, "server_overall", report.server_latency.overall,
      first_latency_field);
  append_latency_summary_json(
      json, "server_preprocess", report.server_latency.preprocess,
      first_latency_field);
  append_latency_summary_json(
      json, "server_queue", report.server_latency.queue, first_latency_field);
  append_latency_summary_json(
      json, "server_batching", report.server_latency.batching,
      first_latency_field);
  append_latency_summary_json(
      json, "server_submit", report.server_latency.submit, first_latency_field);
  append_latency_summary_json(
      json, "server_scheduling", report.server_latency.scheduling,
      first_latency_field);
  append_latency_summary_json(
      json, "server_codelet", report.server_latency.codelet,
      first_latency_field);
  append_latency_summary_json(
      json, "server_inference", report.server_latency.inference,
      first_latency_field);
  append_latency_summary_json(
      json, "server_callback", report.server_latency.callback,
      first_latency_field);
  append_latency_summary_json(
      json, "server_postprocess", report.server_latency.postprocess,
      first_latency_field);
  append_latency_summary_json(
      json, "server_job_total", report.server_latency.job_total,
      first_latency_field);
  append_latency_summary_json(
      json, "request", report.request_latency, first_latency_field);
  append_latency_summary_json(
      json, "response", report.response_latency, first_latency_field);
  append_latency_summary_json(
      json, "client_overhead", report.client_overhead_latency,
      first_latency_field);
  json += "\n  }\n";
  json += "}\n";

  stream << json;
  if (!stream.good()) {
    log_error(
        std::format("Failed to write summary JSON file '{}'", path.string()));
    return false;
  }
  return true;
}

void
InferenceClient::log_latency_summary() const
{
  const auto report = summary();
  if (!report.roundtrip_latency.has_value()) {
    return;
  }

  const auto sample_count = report.response_count;
  std::string stats_msg = std::format(
      "Processed {} inference responses. Latency statistics (ms):",
      sample_count);

  const auto append_stats =
      [&stats_msg](
          const char* label, const std::optional<LatencySummary>& latencies) {
        if (!latencies.has_value()) {
          return;
        }

        stats_msg += std::format(
            "\n  - {}: mean={:.3f}, p50={:.3f}, p85={:.3f}, p95={:.3f}, "
            "p100={:.3f}",
            label, latencies->mean_ms, latencies->p50_ms, latencies->p85_ms,
            latencies->p95_ms, latencies->p100_ms);
      };

  append_stats("latency", report.roundtrip_latency);
  append_stats("server overall", report.server_latency.overall);
  append_stats("preprocess", report.server_latency.preprocess);
  append_stats("queue", report.server_latency.queue);
  append_stats("batching", report.server_latency.batching);
  append_stats("submit", report.server_latency.submit);
  append_stats("scheduling", report.server_latency.scheduling);
  append_stats("codelet", report.server_latency.codelet);
  append_stats("inference", report.server_latency.inference);
  append_stats("callback", report.server_latency.callback);
  append_stats("postprocess", report.server_latency.postprocess);
  append_stats("job_total", report.server_latency.job_total);
  append_stats("request_latency", report.request_latency);
  append_stats("response_latency", report.response_latency);
  append_stats("client_overhead", report.client_overhead_latency);

  if (report.throughput_rps.has_value() && report.elapsed_seconds.has_value()) {
    stats_msg += std::format(
        "\n  - throughput: {:.3f} inferences/s ({} inferences over {:.3f} s)",
        *report.throughput_rps, report.inference_count,
        *report.elapsed_seconds);
  } else if (report.inference_count > 0) {
    stats_msg += std::format(
        "\n  - throughput: {} inferences (elapsed time too small to compute "
        "rate)",
        report.inference_count);
  }

  log_info(verbosity_, stats_msg);
}

auto
InferenceClient::determine_inference_count(const ClientConfig& cfg)
    -> std::size_t
{
  if (cfg.inputs.empty()) {
    return 1;
  }

  std::optional<int64_t> batch_dim;
  for (const auto& input : cfg.inputs) {
    if (input.shape.empty()) {
      continue;
    }

    const int64_t current_dim = input.shape.front();
    if (current_dim <= 0) {
      continue;
    }

    if (!batch_dim.has_value()) {
      batch_dim = current_dim;
      continue;
    }

    if (*batch_dim != current_dim) {
      log_warning(std::format(
          "Inconsistent batch dimension across inputs ({} vs {}). Using {}.",
          *batch_dim, current_dim, *batch_dim));
    }
  }

  if (!batch_dim.has_value()) {
    return 1;
  }

  return static_cast<std::size_t>(*batch_dim);
}

auto
InferenceClient::ServerIsLive() -> bool
{
  const inference::ServerLiveRequest request;
  inference::ServerLiveResponse response;
  grpc::ClientContext context;

  grpc::Status status = stub_->ServerLive(&context, request, &response);

  if (!status.ok()) {
    log_error(std::format("RPC failed: {}", status.error_message()));
    return false;
  }

  log_info(
      verbosity_,
      std::format("Server live: {}", response.live() ? "true" : "false"));
  return response.live();
}

auto
InferenceClient::ServerIsReady() -> bool
{
  const inference::ServerReadyRequest request;
  inference::ServerReadyResponse response;
  grpc::ClientContext context;

  grpc::Status status = stub_->ServerReady(&context, request, &response);

  if (!status.ok()) {
    log_error(std::format("RPC failed: {}", status.error_message()));
    return false;
  }

  log_info(
      verbosity_,
      std::format("Server ready: {}", response.ready() ? "true" : "false"));
  return response.ready();
}

auto
InferenceClient::ModelIsReady(const ModelId& model) -> bool
{
  inference::ModelReadyRequest request;
  request.set_name(model.name);
  request.set_version(model.version);
  inference::ModelReadyResponse response;
  grpc::ClientContext context;

  grpc::Status status = stub_->ModelReady(&context, request, &response);

  if (!status.ok()) {
    log_error(std::format("RPC failed: {}", status.error_message()));
    return false;
  }

  log_info(
      verbosity_,
      std::format("Model ready: {}", response.ready() ? "true" : "false"));
  return response.ready();
}

void
InferenceClient::AsyncModelInfer(
    const std::vector<torch::Tensor>& tensors, const ClientConfig& cfg,
    std::optional<OutputSummary> expected_outputs)
{
  const int current_id = next_request_id_++;

  auto call = std::make_unique<AsyncClientCall>();
  call->request_id = current_id;
  call->start_time = std::chrono::system_clock::now();
  call->inference_count = determine_inference_count(cfg);
  call->expected_outputs = std::move(expected_outputs);
  ++total_requests_sent_;

  if (!first_request_time_) {
    first_request_time_ = call->start_time;
  }

  if (should_log(VerbosityLevel::Stats, verbosity_)) {
    log_stats(verbosity_, std::format("Sending request ID: {}", current_id));
  }

  inference::ModelInferRequest request;
  request.set_model_name(cfg.model_name);
  request.set_model_version(cfg.model_version);
  request.set_client_send_ms(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          call->start_time.time_since_epoch())
          .count());

  if (tensors.size() != cfg.inputs.size()) {
    auto msg = std::format(
        "Mismatched number of input tensors: expected {}, got {}",
        cfg.inputs.size(), tensors.size());
    log_error(msg);
    throw std::invalid_argument(msg);
  }

  for (size_t i = 0; i < cfg.inputs.size(); ++i) {
    const auto& in_cfg = cfg.inputs[i];
    torch::Tensor tensor = tensors.at(i);

    if (tensor.scalar_type() != in_cfg.type) {
      auto msg = std::format(
          "Unsupported tensor type for input {}: expected {}, got {}",
          in_cfg.name, scalar_type_to_string(in_cfg.type),
          scalar_type_to_string(tensor.scalar_type()));
      log_error(msg);
      throw std::invalid_argument(msg);
    }

    if (!tensor.device().is_cpu() || !tensor.is_contiguous()) {
      log_info(
          verbosity_,
          std::format(
              "Input tensor {} not on CPU or non-contiguous, converting",
              in_cfg.name));
      tensor = tensor.cpu().contiguous();
    }
    auto* input = request.add_inputs();
    input->set_name(in_cfg.name);
    input->set_datatype(scalar_type_to_string(in_cfg.type));
    for (auto dim : tensor.sizes()) {
      input->add_shape(dim);
    }
    auto flat = tensor.view({-1});
    request.add_raw_input_contents()->assign(
        static_cast<const char*>(flat.data_ptr()),
        flat.numel() * flat.element_size());
  }

  call->response_reader = stub_->AsyncModelInfer(&call->context, request, &cq_);
  call->response_reader->Finish(&call->reply, &call->status, call.get());
  [[maybe_unused]] auto* released_call = call.release();
}

void
InferenceClient::validate_server_response(const AsyncClientCall& call) const
{
  if (!call.expected_outputs.has_value()) {
    return;
  }

  const auto& expected = *call.expected_outputs;
  if (expected.empty()) {
    return;
  }

  const int server_outputs = call.reply.outputs_size();
  if (server_outputs == 0) {
    log_warning(std::format(
        "Request {}: server response does not contain outputs; skipping "
        "validation.",
        call.request_id));
    return;
  }

  const int raw_count = call.reply.raw_output_contents_size();
  const std::size_t outputs_to_check =
      std::min<std::size_t>(expected.size(), server_outputs);

  for (std::size_t idx = 0; idx < outputs_to_check; ++idx) {
    const auto& expected_values = expected[idx];
    if (expected_values.empty()) {
      continue;
    }

    if (static_cast<int>(idx) >= raw_count) {
      log_warning(std::format(
          "Request {} output {} missing raw contents; skipping validation.",
          call.request_id, idx));
      continue;
    }

    const auto& tensor_meta = call.reply.outputs(static_cast<int>(idx));
    const auto& raw = call.reply.raw_output_contents(static_cast<int>(idx));
    const std::string_view raw_view(raw.data(), raw.size());

    std::vector<double> decoded;
    try {
      decoded =
          decode_output_values(tensor_meta, raw_view, expected_values.size());
    }
    catch (const std::exception& e) {
      log_warning(std::format(
          "Request {} output {}: failed to decode server output: {}",
          call.request_id, idx, e.what()));
      continue;
    }

    if (decoded.size() != expected_values.size()) {
      log_warning(std::format(
          "Request {} output {}: expected {} values, decoded {}",
          call.request_id, idx, expected_values.size(), decoded.size()));
      continue;
    }

    bool mismatch = false;
    double max_diff = 0.0;
    std::size_t max_diff_idx = 0;
    double expected_at_max = 0.0;
    double decoded_at_max = 0.0;
    for (std::size_t value_idx = 0; value_idx < decoded.size(); ++value_idx) {
      const double diff =
          std::abs(decoded[value_idx] - expected_values[value_idx]);
      if (diff > max_diff) {
        max_diff = diff;
        max_diff_idx = value_idx;
        expected_at_max = expected_values[value_idx];
        decoded_at_max = decoded[value_idx];
      }
    }

    if (max_diff > kOutputComparisonTolerance) {
      mismatch = true;
      log_warning(std::format(
          "Request {} output {} max mismatch at value {}: expected {:.6f}, got "
          "{:.6f} (Δ={:.6f})",
          call.request_id, idx, max_diff_idx, expected_at_max, decoded_at_max,
          max_diff));
    }

    if (!mismatch) {
      log_trace(
          verbosity_, std::format(
                          "Request {} output {} validated on {} values",
                          call.request_id, idx, decoded.size()));
    }
  }

  if (static_cast<std::size_t>(server_outputs) < expected.size()) {
    log_warning(std::format(
        "Request {}: server returned {} outputs but {} were available for "
        "validation.",
        call.request_id, server_outputs, expected.size()));
  }
}

void
InferenceClient::AsyncCompleteRpc()
{
  void* got_tag = nullptr;
  bool call_ctx = false;
  while (cq_.Next(&got_tag, &call_ctx)) {
    if (got_tag == nullptr) {
      log_warning("Received invalid RPC completion, exiting CQ loop");
      break;
    }

    auto call = std::unique_ptr<AsyncClientCall>(
        static_cast<AsyncClientCall*>(got_tag));
    auto end = std::chrono::system_clock::now();
    if (!call_ctx) {
      auto recv_time_str = time_utils::format_timestamp(end);
      log_warning(std::format(
          "Request ID {} completion not ok at {}; treating as failure",
          call->request_id, recv_time_str));
      ++rejected_requests_;
      continue;
    }
    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end - call->start_time)
                       .count();
    const auto end_ms = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            end.time_since_epoch())
            .count());

    if (call->status.ok()) {
      const auto start_ms = static_cast<int64_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              call->start_time.time_since_epoch())
              .count());
      const auto server_receive_ms =
          static_cast<int64_t>(call->reply.server_receive_ms());
      const auto server_send_ms =
          static_cast<int64_t>(call->reply.server_send_ms());
      int64_t request_latency_ms = server_receive_ms - start_ms;
      int64_t response_latency_ms = end_ms - server_send_ms;
      if (request_latency_ms < 0) {
        request_latency_ms = 0;
      }
      if (response_latency_ms < 0) {
        response_latency_ms = 0;
      }
      const auto server_total_ms = call->reply.server_total_ms();
      const auto queue_ms = call->reply.server_queue_ms();
      const auto batch_ms = call->reply.server_batch_ms();
      const auto submit_ms = call->reply.server_submit_ms();
      const auto scheduling_ms = call->reply.server_scheduling_ms();
      const auto codelet_ms = call->reply.server_codelet_ms();
      const auto inference_ms = call->reply.server_inference_ms();
      const auto callback_ms = call->reply.server_callback_ms();
      const auto preprocess_ms = call->reply.server_preprocess_ms();
      const auto postprocess_ms = call->reply.server_postprocess_ms();
      const auto overall_ms = call->reply.server_overall_ms();

      const double accounted_roundtrip_ms =
          static_cast<double>(request_latency_ms) +
          static_cast<double>(overall_ms) +
          static_cast<double>(response_latency_ms);
      double client_overhead_ms =
          static_cast<double>(latency) - accounted_roundtrip_ms;
      if (client_overhead_ms < 0.0) {
        client_overhead_ms = 0.0;
      }

      const LatencySample sample{
          static_cast<double>(latency),
          static_cast<double>(overall_ms),
          static_cast<double>(preprocess_ms),
          static_cast<double>(queue_ms),
          static_cast<double>(batch_ms),
          static_cast<double>(submit_ms),
          static_cast<double>(scheduling_ms),
          static_cast<double>(codelet_ms),
          static_cast<double>(inference_ms),
          static_cast<double>(callback_ms),
          static_cast<double>(postprocess_ms),
          static_cast<double>(server_total_ms),
          static_cast<double>(request_latency_ms),
          static_cast<double>(response_latency_ms),
          client_overhead_ms};
      record_latency(sample);

      if (should_log(VerbosityLevel::Stats, verbosity_)) {
        auto sent_time_str = time_utils::format_timestamp(call->start_time);
        auto recv_time_str = time_utils::format_timestamp(end);
        log_stats(
            verbosity_,
            std::format(
                "Request ID {} sent at {} received at {} latency: {} "
                "ms (server overall: {:.3f} ms | preprocess: {:.3f} ms, queue: "
                "{:.3f} ms, batching: {:.3f} ms, submit: {:.3f} ms, "
                "scheduling: "
                "{:.3f} ms, codelet: {:.3f} ms, inference: {:.3f} ms, "
                "callback: "
                "{:.3f} ms, postprocess: {:.3f} ms, job_total: {:.3f} ms), "
                "request_latency: {} ms, response_latency: {} ms, "
                "client_overhead: {:.3f} ms",
                call->request_id, sent_time_str, recv_time_str, latency,
                overall_ms, preprocess_ms, queue_ms, batch_ms, submit_ms,
                scheduling_ms, codelet_ms, inference_ms, callback_ms,
                postprocess_ms, server_total_ms, request_latency_ms,
                response_latency_ms, client_overhead_ms));
      }

      validate_server_response(*call);
      total_inference_count_ += call->inference_count;
      last_response_time_ = end;
      ++success_requests_;
    } else {
      auto recv_time_str = time_utils::format_timestamp(end);
      log_error(std::format(
          "Request ID {} failed at {}: {}", call->request_id, recv_time_str,
          call->status.error_message()));
      ++rejected_requests_;
    }
  }

  log_latency_summary();
  log_request_totals();
}

void
InferenceClient::Shutdown()
{
  cq_.Shutdown();
}

void
InferenceClient::log_request_totals() const
{
  const std::size_t completed = success_requests_ + rejected_requests_;
  const std::size_t total = std::max(total_requests_sent_, completed);
  log_info(
      verbosity_,
      std::format(
          "Requests summary: {} handled ({} ok, {} rejected) out of {} sent",
          completed, success_requests_, rejected_requests_, total));
}
}  // namespace starpu_server
