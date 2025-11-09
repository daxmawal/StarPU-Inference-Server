#include "inference_client.hpp"

#include <cstdint>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>

#include "core/latency_statistics.hpp"
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
  std::chrono::high_resolution_clock::time_point start_time;
  int request_id = 0;
  std::size_t inference_count = 1;
};

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

void
InferenceClient::log_latency_summary() const
{
  if (latency_records_.empty()) {
    return;
  }

  const auto sample_count = latency_records_.roundtrip_ms.size();
  std::string stats_msg = std::format(
      "Processed {} inference responses. Latency statistics (ms):",
      sample_count);

  const auto append_stats =
      [&stats_msg](const char* label, const std::vector<double>& latencies) {
        if (latencies.empty()) {
          return;
        }

        if (const auto stats = compute_latency_statistics(latencies)) {
          stats_msg += std::format(
              "\n  - {}: mean={:.3f}, p50={:.3f}, p85={:.3f}, p95={:.3f}, "
              "p100={:.3f}",
              label, stats->mean, stats->p50, stats->p85, stats->p95,
              stats->p100);
        }
      };

  append_stats("latency", latency_records_.roundtrip_ms);
  append_stats("server overall", latency_records_.server_overall_ms);
  append_stats("preprocess", latency_records_.server_preprocess_ms);
  append_stats("queue", latency_records_.server_queue_ms);
  append_stats("batching", latency_records_.server_batch_ms);
  append_stats("submit", latency_records_.server_submit_ms);
  append_stats("scheduling", latency_records_.server_scheduling_ms);
  append_stats("codelet", latency_records_.server_codelet_ms);
  append_stats("inference", latency_records_.server_inference_ms);
  append_stats("callback", latency_records_.server_callback_ms);
  append_stats("postprocess", latency_records_.server_postprocess_ms);
  append_stats("job_total", latency_records_.server_job_total_ms);
  append_stats("request_latency", latency_records_.request_latency_ms);
  append_stats("response_latency", latency_records_.response_latency_ms);
  append_stats("client_overhead", latency_records_.client_overhead_ms);

  if (first_request_time_.has_value() && last_response_time_.has_value() &&
      total_inference_count_ > 0) {
    const auto elapsed_seconds =
        std::chrono::duration<double>(
            *last_response_time_ - *first_request_time_)
            .count();
    if (elapsed_seconds > 0.0) {
      const double throughput =
          static_cast<double>(total_inference_count_) / elapsed_seconds;
      const auto batch_size =
          last_batch_size_.has_value() ? *last_batch_size_ : std::size_t{1};
      stats_msg += std::format(
          "\n  - throughput: {:.3f} inferences/s ({} inferences over {:.3f} s)",
          throughput, total_inference_count_, elapsed_seconds);
    } else {
      stats_msg += std::format(
          "\n  - throughput: {} inferences (elapsed time too small to compute "
          "rate)",
          total_inference_count_);
    }
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
    const std::vector<torch::Tensor>& tensors, const ClientConfig& cfg)
{
  const int current_id = next_request_id_++;

  auto call = std::make_unique<AsyncClientCall>();
  call->request_id = current_id;
  call->start_time = std::chrono::high_resolution_clock::now();
  call->inference_count = determine_inference_count(cfg);
  last_batch_size_ = call->inference_count;

  if (!first_request_time_) {
    first_request_time_ = call->start_time;
  }

  log_info(verbosity_, std::format("Sending request ID: {}", current_id));

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
    for (auto dim : in_cfg.shape) {
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
InferenceClient::AsyncCompleteRpc()
{
  void* got_tag = nullptr;
  bool call_ctx = false;
  while (cq_.Next(&got_tag, &call_ctx)) {
    if (!call_ctx) {
      log_warning("Received invalid RPC completion, exiting CQ loop");
      break;
    }

    auto call = std::unique_ptr<AsyncClientCall>(
        static_cast<AsyncClientCall*>(got_tag));
    auto end = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end - call->start_time)
                       .count();
    const auto end_ms = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            end.time_since_epoch())
            .count());

    auto sent_time_str = time_utils::format_timestamp(call->start_time);
    auto recv_time_str = time_utils::format_timestamp(end);

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

      log_info(
          verbosity_,
          std::format(
              "Request ID {} sent at {} received at {} latency: {} "
              "ms (server overall: {:.3f} ms | preprocess: {:.3f} ms, queue: "
              "{:.3f} ms, batching: {:.3f} ms, submit: {:.3f} ms, scheduling: "
              "{:.3f} ms, codelet: {:.3f} ms, inference: {:.3f} ms, callback: "
              "{:.3f} ms, postprocess: {:.3f} ms, job_total: {:.3f} ms), "
              "request_latency: {} ms, response_latency: {} ms, "
              "client_overhead: {:.3f} ms",
              call->request_id, sent_time_str, recv_time_str, latency,
              overall_ms, preprocess_ms, queue_ms, batch_ms, submit_ms,
              scheduling_ms, codelet_ms, inference_ms, callback_ms,
              postprocess_ms, server_total_ms, request_latency_ms,
              response_latency_ms, client_overhead_ms));

      total_inference_count_ += call->inference_count;
      last_response_time_ = end;
    } else {
      log_error(std::format(
          "Request ID {} failed at {}: {}", call->request_id, recv_time_str,
          call->status.error_message()));
    }
  }

  log_latency_summary();
}

void
InferenceClient::Shutdown()
{
  cq_.Shutdown();
}
}  // namespace starpu_server
