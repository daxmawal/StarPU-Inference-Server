#include "inference_service.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "core/inference_runner.hpp"
#include "monitoring/metrics.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/client_utils.hpp"
#include "utils/datatype_utils.hpp"
#include "utils/logger.hpp"
#include "utils/nvtx.hpp"

namespace starpu_server {
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using inference::ModelInferRequest;
using inference::ModelInferResponse;
using inference::ModelReadyRequest;
using inference::ModelReadyResponse;
using inference::ServerLiveRequest;
using inference::ServerLiveResponse;
using inference::ServerReadyRequest;
using inference::ServerReadyResponse;

auto
compute_thread_count_from(unsigned concurrency) -> std::size_t
{
  if (concurrency == 0U) {
    return kDefaultGrpcThreads;
  }
  return std::clamp<std::size_t>(concurrency, kMinGrpcThreads, kMaxGrpcThreads);
}

namespace {

// Congestion detection parameters:
// Congestion is detected when the arrival rate reaches 95% of measured
// throughput. This allows clients to be aware when the system is near capacity.
constexpr double kCongestionEnterRatio = 0.95;

// Congestion is cleared when arrival rate falls below 90% of measured
// throughput. This hysteresis prevents rapid oscillation between
// detected/cleared states.
constexpr double kCongestionClearRatio = 0.90;

// Sliding window duration for measuring request arrival rate.
// Tracks requests arriving within the last 1 second.
constexpr auto kArrivalWindow = std::chrono::seconds(1);

// How often the congestion monitor thread checks for stale arrival windows
// and logs congestion state transitions.
constexpr auto kCongestionMonitorPeriod = std::chrono::milliseconds(200);

auto
parse_input_dtype(
    const inference::ModelInferRequest::InferInputTensor& input,
    at::ScalarType expected, at::ScalarType& out) -> Status
{
  try {
    out = datatype_to_scalar_type(input.datatype());
  }
  catch (const std::invalid_argument& e) {
    return {grpc::StatusCode::INVALID_ARGUMENT, e.what()};
  }

  if (out != expected) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT, "Input tensor datatype mismatch"};
  }

  return Status::OK;
}

// Helper: checks if shape tail matches expected tail at given offsets
auto
shape_tails_match(
    const std::vector<int64_t>& shape, const std::vector<int64_t>& expected,
    int64_t shape_offset, int64_t expected_offset) -> bool
{
  const auto rank = static_cast<int64_t>(shape.size());
  const auto expected_rank = static_cast<int64_t>(expected.size());

  if (rank - shape_offset != expected_rank - expected_offset) {
    return false;
  }
  for (int64_t idx = 0; idx < rank - shape_offset; ++idx) {
    if (shape[static_cast<size_t>(shape_offset + idx)] !=
        expected[static_cast<size_t>(expected_offset + idx)]) {
      return false;
    }
  }
  return true;
}

// Helper: validates batch size is positive and doesn't exceed configured max
auto
validate_batch_size(int64_t batch_size, int max_batch_size) -> Status
{
  if (batch_size <= 0) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "Input tensor shape does not match configured dimensions or batch "
        "limits (batch size must be positive)"};
  }
  if (batch_size > max_batch_size) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        std::format(
            "Input tensor shape does not match configured "
            "dimensions or batch limits (batch size {} exceeds "
            "configured max of {})",
            batch_size, max_batch_size)};
  }
  return Status::OK;
}

auto
validate_configured_shape(
    const std::vector<int64_t>& shape, const std::vector<int64_t>& expected,
    bool batching_allowed, int max_batch_size) -> Status
{
  const auto rank = static_cast<int64_t>(shape.size());

  // Case 1: Batching not allowed - shape must exactly match expected
  if (!batching_allowed) {
    if (shape_tails_match(shape, expected, 0, 0)) {
      return Status::OK;
    }
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "Input tensor shape does not match configured dimensions or batch "
        "limits"};
  }

  // Case 2: Empty shape is invalid
  if (rank == 0) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "Input tensor shape does not match configured dimensions or batch "
        "limits"};
  }

  // Case 3: Shape matches exactly (no batching)
  if (shape_tails_match(shape, expected, 0, 0)) {
    return Status::OK;
  }

  // Case 4: Shape with batch dimension at front
  if (rank >= 1 && shape_tails_match(shape, expected, 1, 1)) {
    const int64_t batch_size = shape.front();
    return validate_batch_size(batch_size, max_batch_size);
  }

  // Case 5: Shape with batch dimension, expected without batch dimension
  if (rank >= 1 && shape_tails_match(shape, expected, 1, 0)) {
    const int64_t batch_size = shape.front();
    return validate_batch_size(batch_size, max_batch_size);
  }

  return {
      grpc::StatusCode::INVALID_ARGUMENT,
      "Input tensor shape does not match configured dimensions or batch "
      "limits"};
}

auto
checked_mul(size_t lhs, size_t rhs) -> std::optional<size_t>
{
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    return std::nullopt;
  }
  return lhs * rhs;
}

// Converts gRPC input tensor data to PyTorch tensor, managing data lifetime.
//
// This function handles a complex data ownership scenario:
// - The raw tensor data comes from the gRPC request (managed by request_guard)
// - We need to create a PyTorch tensor that references this data
// - The tensor must be valid even after the gRPC processing completes
//
// Memory management strategy:
// 1. Create an alias shared_ptr that points to the raw data but holds a
// reference
//    to request_guard (keeping the gRPC request alive while tensor is in use)
// 2. PyTorch's from_blob requires a custom deleter to notify when tensor is
// freed
// 3. The deleter holds a reference to the alias, which keeps request_guard
// alive
// 4. When the tensor is destroyed, deleter runs and releases the alias
//
// The caller can optionally capture 'keep_alive' to hold an explicit reference
// to the alias, ensuring the data lives as long as needed.
auto
convert_input_to_tensor(
    const ModelInferRequest::InferInputTensor& input, const std::string& raw,
    at::ScalarType dtype,
    const std::shared_ptr<const ModelInferRequest>& request_guard,
    torch::Tensor& tensor, std::shared_ptr<const void>* keep_alive) -> Status
{
  std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
  auto options = torch::TensorOptions().dtype(dtype);

  std::optional<size_t> expected = element_size(dtype);
  for (const auto dim : input.shape()) {
    if (dim <= 0) {
      return {
          grpc::StatusCode::INVALID_ARGUMENT,
          "Input tensor shape contains non-positive dimension"};
    }
    expected = checked_mul(*expected, static_cast<size_t>(dim));
    if (!expected) {
      return {
          grpc::StatusCode::INVALID_ARGUMENT,
          "Input tensor shape is too large"};
    }
  }
  if (*expected != raw.size()) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "Raw input size does not match tensor size"};
  }

  // Create alias: a shared_ptr that points to the raw data but holds a
  // reference to request_guard to keep the gRPC message alive
  const auto raw_span = std::span<const char>(raw.data(), raw.size());
  const auto byte_span = std::as_bytes(raw_span);
  const auto* byte_data = byte_span.data();
  auto alias = std::shared_ptr<const std::byte>(request_guard, byte_data);
  auto holder = std::const_pointer_cast<std::byte>(alias);

  // Custom deleter that captures the alias. When PyTorch releases the tensor,
  // the deleter runs and allows alias (and thus request_guard) to be freed.
  auto deleter = [holder](void* /*unused*/) mutable {
    auto keep = holder;
    keep.reset();
  };

  std::byte* tensor_data = holder.get();
  tensor = torch::from_blob(tensor_data, shape, deleter, options);
  if (keep_alive != nullptr) {
    *keep_alive = alias;
  }
  return Status::OK;
}

auto
fill_output_tensor(
    ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs) -> Status
{
  for (size_t idx = 0; idx < outputs.size(); ++idx) {
    const auto& original = outputs[idx];
    torch::Tensor out = original;
    if (!original.device().is_cpu()) {
      out = original.to(torch::kCPU);
    }
    auto* out_tensor = reply->add_outputs();
    out_tensor->set_name(std::format("output{}", idx));
    out_tensor->set_datatype(scalar_type_to_datatype(out.scalar_type()));
    for (const auto dim : out.sizes()) {
      out_tensor->add_shape(dim);
    }

    auto flat = out.contiguous().view({-1});
    const auto total_bytes =
        checked_mul(static_cast<size_t>(flat.numel()), flat.element_size());
    if (!total_bytes.has_value()) {
      return {
          grpc::StatusCode::INVALID_ARGUMENT, "Output tensor size overflow"};
    }
    reply->add_raw_output_contents()->assign(
        static_cast<const char*>(flat.data_ptr()), *total_bytes);
  }
  return Status::OK;
}
}  // namespace

InferenceServiceImpl::InferenceServiceImpl(
    InferenceQueue* queue, const std::vector<torch::Tensor>* reference_outputs,
    std::vector<at::ScalarType> expected_input_types,
    std::vector<std::vector<int64_t>> expected_input_dims, int max_batch_size,
    std::string default_model_name, double measured_throughput, bool use_cuda,
    double measured_throughput_cpu)
    : queue_(queue), reference_outputs_(reference_outputs),
      expected_input_types_(std::move(expected_input_types)),
      expected_input_dims_(std::move(expected_input_dims)),
      max_batch_size_(max_batch_size),
      default_model_name_(std::move(default_model_name)), use_cuda_(use_cuda),
      measured_throughput_(
          use_cuda ? measured_throughput : measured_throughput_cpu),
      congestion_threshold_(
          (use_cuda ? measured_throughput : measured_throughput_cpu) *
          kCongestionEnterRatio),
      congestion_clear_threshold_(
          (use_cuda ? measured_throughput : measured_throughput_cpu) *
          kCongestionClearRatio)
{
  start_congestion_monitor();
}

InferenceServiceImpl::InferenceServiceImpl(
    InferenceQueue* queue, const std::vector<torch::Tensor>* reference_outputs,
    std::vector<at::ScalarType> expected_input_types,
    std::string default_model_name, double measured_throughput, bool use_cuda,
    double measured_throughput_cpu)
    : InferenceServiceImpl(
          queue, reference_outputs, std::move(expected_input_types), {}, 0,
          std::move(default_model_name), measured_throughput, use_cuda,
          measured_throughput_cpu)
{
}

auto
InferenceServiceImpl::ServerLive(
    ServerContext* /*context*/, const ServerLiveRequest* /*request*/,
    ServerLiveResponse* reply) -> Status
{
  reply->set_live(true);
  return Status::OK;
}

auto
InferenceServiceImpl::ServerReady(
    ServerContext* /*context*/, const ServerReadyRequest* /*request*/,
    ServerReadyResponse* reply) -> Status
{
  reply->set_ready(true);
  return Status::OK;
}

auto
InferenceServiceImpl::ModelReady(
    ServerContext* /*context*/, const ModelReadyRequest* /*request*/,
    ModelReadyResponse* reply) -> Status
{
  reply->set_ready(true);
  return Status::OK;
}

auto
InferenceServiceImpl::validate_and_convert_inputs(
    const ModelInferRequest* request, std::vector<torch::Tensor>& inputs,
    std::vector<std::shared_ptr<const void>>* input_lifetimes) -> Status
{
  if (request->inputs_size() !=
      static_cast<int>(expected_input_types_.size())) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        std::format(
            "Expected {} input tensors but received {}",
            expected_input_types_.size(), request->inputs_size())};
  }
  if (request->raw_input_contents_size() != request->inputs_size()) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "Number of raw inputs does not match number of input tensors"};
  }

  auto request_guard = std::shared_ptr<const ModelInferRequest>(
      request, [](const ModelInferRequest*) {});

  inputs.reserve(request->inputs_size());
  if (input_lifetimes != nullptr) {
    input_lifetimes->clear();
    input_lifetimes->reserve(request->inputs_size());
  }
  for (int i = 0; i < request->inputs_size(); ++i) {
    const auto& input = request->inputs(i);
    const auto& raw = request->raw_input_contents(i);

    at::ScalarType dtype = at::kFloat;
    Status status = parse_input_dtype(input, expected_input_types_[i], dtype);
    if (!status.ok()) {
      return status;
    }

    torch::Tensor tensor;
    std::shared_ptr<const void> tensor_guard;
    status = convert_input_to_tensor(
        input, raw, dtype, request_guard, tensor,
        input_lifetimes != nullptr ? &tensor_guard : nullptr);
    if (!status.ok()) {
      return status;
    }

    if (static_cast<size_t>(i) < expected_input_dims_.size()) {
      const auto& expected_dims = expected_input_dims_[static_cast<size_t>(i)];
      std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
      const bool batching_allowed = (max_batch_size_ > 0);
      status = validate_configured_shape(
          shape, expected_dims, batching_allowed, max_batch_size_);
      if (!status.ok()) {
        return status;
      }
    }

    inputs.push_back(std::move(tensor));
    if (input_lifetimes != nullptr) {
      input_lifetimes->push_back(std::move(tensor_guard));
    }
  }
  return Status::OK;
}

auto
InferenceServiceImpl::build_latency_breakdown(
    const detail::TimingInfo& info, double latency_ms) -> LatencyBreakdown
{
  const auto base = detail::compute_latency_breakdown(info, latency_ms);
  LatencyBreakdown breakdown{};
  breakdown.queue_ms = base.queue_ms;
  breakdown.batch_ms = base.batch_ms;
  breakdown.submit_ms = base.submit_ms;
  breakdown.scheduling_ms = base.scheduling_ms;
  breakdown.codelet_ms = base.codelet_ms;
  breakdown.inference_ms = base.inference_ms;
  breakdown.callback_ms = base.callback_ms;
  breakdown.total_ms = base.total_ms;
  return breakdown;
}

auto
InferenceServiceImpl::resolve_model_name(std::string model_name) const
    -> std::string
{
  if (default_model_name_.empty()) {
    return model_name;
  }
  if (!model_name.empty() && model_name != default_model_name_) {
    log_warning(std::format(
        "Client requested model '{}' but server is configured for '{}'; "
        "defaulting to server model",
        model_name, default_model_name_));
  }
  return default_model_name_;
}

void
InferenceServiceImpl::start_congestion_monitor()
{
  if (measured_throughput_ <= 0.0) {
    return;
  }

  congestion_monitor_thread_ =
      std::jthread([this](const std::stop_token& stop_token) {
        while (!stop_token.stop_requested()) {
          std::chrono::high_resolution_clock::time_point start_time{};
          std::chrono::high_resolution_clock::time_point end_time{};
          bool cleared = false;
          {
            const auto now = std::chrono::high_resolution_clock::now();
            std::scoped_lock lock(congestion_mutex_);
            const bool stale_window =
                congestion_active_ &&
                (last_arrival_time_ ==
                     std::chrono::high_resolution_clock::time_point{} ||
                 now - last_arrival_time_ > kArrivalWindow);
            if (stale_window) {
              congestion_active_ = false;
              start_time = congestion_start_time_;
              end_time = now;
              congestion_start_time_ =
                  std::chrono::high_resolution_clock::time_point{};
              recent_arrivals_.clear();
              const auto* const device_name = use_cuda_ ? "GPUs" : "CPUs";
              log_warning(std::format(
                  "[Congestion] {} congestion cleared: arrival rate {:.2f} "
                  "req/s is below {:.2f} infer/s",
                  device_name, 0.0, congestion_clear_threshold_));
              cleared = true;
            }
          }
          if (cleared) {
            if (start_time ==
                std::chrono::high_resolution_clock::time_point{}) {
              start_time = end_time - kArrivalWindow;
            }
            BatchingTraceLogger::instance().log_congestion_span(
                BatchingTraceLogger::CongestionSpanArgs{
                    .start_time = start_time,
                    .end_time = end_time,
                    .measured_throughput = measured_throughput_,
                    .enter_threshold = congestion_threshold_,
                    .clear_threshold = congestion_clear_threshold_,
                });
          }
          std::this_thread::sleep_for(kCongestionMonitorPeriod);
        }
      });
}

void
InferenceServiceImpl::record_request_arrival(
    std::chrono::high_resolution_clock::time_point arrival_time)
{
  if (measured_throughput_ <= 0.0) {
    return;
  }

  bool entered = false;
  bool cleared = false;
  double arrival_rate = 0.0;
  double enter_threshold = congestion_threshold_;
  double clear_threshold = congestion_clear_threshold_;
  std::chrono::high_resolution_clock::time_point start_time{};

  {
    const std::scoped_lock lock(congestion_mutex_);
    last_arrival_time_ = arrival_time;
    recent_arrivals_.push_back(arrival_time);
    const auto cutoff = arrival_time - kArrivalWindow;
    while (!recent_arrivals_.empty() && recent_arrivals_.front() < cutoff) {
      recent_arrivals_.pop_front();
    }

    if (recent_arrivals_.empty()) {
      return;
    }

    const double window_seconds = std::max(
        std::chrono::duration<double>(kArrivalWindow).count(),
        std::chrono::duration<double>(
            recent_arrivals_.back() - recent_arrivals_.front())
            .count());
    if (window_seconds <= 0.0) {
      return;
    }

    arrival_rate =
        static_cast<double>(recent_arrivals_.size()) / window_seconds;

    if (!congestion_active_ && arrival_rate >= congestion_threshold_) {
      congestion_active_ = true;
      congestion_start_time_ = arrival_time;
      entered = true;
    } else if (
        congestion_active_ && arrival_rate < congestion_clear_threshold_) {
      congestion_active_ = false;
      start_time = congestion_start_time_;
      congestion_start_time_ = std::chrono::high_resolution_clock::time_point{};
      cleared = true;
    }
  }

  if (entered) {
    const auto* const device_name = use_cuda_ ? "GPUs" : "CPUs";
    log_warning(std::format(
        "[Congestion] {} congestion detected: arrival rate {:.2f} req/s "
        "is near measured throughput {:.2f} infer/s",
        device_name, arrival_rate, measured_throughput_));
    return;
  }

  if (cleared) {
    if (start_time == std::chrono::high_resolution_clock::time_point{}) {
      start_time = arrival_time - kArrivalWindow;
    }
    const auto* const device_name = use_cuda_ ? "GPUs" : "CPUs";
    log_warning(std::format(
        "[Congestion] {} congestion cleared: arrival rate {:.2f} req/s "
        "is below {:.2f} infer/s",
        device_name, arrival_rate, clear_threshold));
    BatchingTraceLogger::instance().log_congestion_span(
        BatchingTraceLogger::CongestionSpanArgs{
            .start_time = start_time,
            .end_time = arrival_time,
            .measured_throughput = measured_throughput_,
            .enter_threshold = enter_threshold,
            .clear_threshold = clear_threshold,
        });
  }
}

auto
InferenceServiceImpl::submit_job_async(
    const std::vector<torch::Tensor>& inputs, AsyncJobCallback on_complete,
    std::vector<std::shared_ptr<const void>> input_lifetimes,
    std::chrono::high_resolution_clock::time_point receive_time,
    std::string model_name) -> Status
{
  auto resolved_model_name = resolve_model_name(std::move(model_name));
  auto job = client_utils::create_job(
      inputs, *reference_outputs_, next_request_id_++,
      std::move(input_lifetimes), receive_time, std::move(resolved_model_name));

  NvtxRange submit_scope("grpc_submit_starpu");

  job->set_on_complete([job_ptr = job, callback = std::move(on_complete)](
                           std::vector<torch::Tensor> outs,
                           double latency_ms) mutable {
    const auto& info = job_ptr->timing_info();
    auto timing = build_latency_breakdown(info, latency_ms);

    if (outs.empty()) {
      callback(
          {grpc::StatusCode::INTERNAL, "Inference failed"}, {}, timing, info);
      return;
    }

    callback(Status::OK, std::move(outs), timing, info);
  });

  bool pushed = false;
  {
    NvtxRange queue_scope("grpc_submit_starpu_queue");
    pushed = queue_->push(job);
    if (pushed) {
      const auto enqueued_now = std::chrono::high_resolution_clock::now();
      job->timing_info().enqueued_time = enqueued_now;
      job->timing_info().last_enqueued_time = enqueued_now;
    }
  }
  if (!pushed) {
    return {grpc::StatusCode::UNAVAILABLE, "Inference queue unavailable"};
  }
  if (auto& tracer = BatchingTraceLogger::instance(); tracer.enabled()) {
    tracer.log_request_enqueued(
        job->get_request_id(), job->model_name(), /*is_warmup=*/false,
        job->timing_info().last_enqueued_time);
  }
  return Status::OK;
}

auto
InferenceServiceImpl::submit_job_and_wait(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs, LatencyBreakdown& breakdown,
    detail::TimingInfo& timing_info,
    std::vector<std::shared_ptr<const void>> input_lifetimes) -> Status
{
  struct JobResult {
    Status status = Status::OK;
    std::vector<torch::Tensor> outputs;
    LatencyBreakdown breakdown;
    detail::TimingInfo timing_info;
  };

  auto result_promise = std::make_shared<std::promise<JobResult>>();
  auto result_future = result_promise->get_future();

  const auto receive_time = std::chrono::high_resolution_clock::now();
  if (Status submit_status = submit_job_async(
          inputs,
          [result_promise](
              Status status, std::vector<torch::Tensor> outs,
              const LatencyBreakdown& cb_breakdown,
              const detail::TimingInfo& cb_timing_info) {
            result_promise->set_value(JobResult{
                std::move(status), std::move(outs), cb_breakdown,
                cb_timing_info});
          },
          std::move(input_lifetimes), receive_time);
      !submit_status.ok()) {
    outputs.clear();
    return submit_status;
  }

  JobResult result = result_future.get();
  if (!result.status.ok()) {
    outputs.clear();
    return result.status;
  }

  outputs = std::move(result.outputs);
  breakdown = result.breakdown;
  timing_info = result.timing_info;
  return Status::OK;
}


InferenceServiceImpl::CallbackHandle::CallbackHandle(
    std::function<void(Status)> callback)
    : callback_(std::move(callback))
{
}

auto
InferenceServiceImpl::CallbackHandle::TryAcquire() -> bool
{
  std::scoped_lock lock(mutex_);
  if (consumed_) {
    return false;
  }
  consumed_ = true;
  return true;
}

void
InferenceServiceImpl::CallbackHandle::Invoke(Status status)
{
  std::function<void(Status)> callback;
  {
    std::scoped_lock lock(mutex_);
    if (!callback_) {
      return;
    }
    consumed_ = true;
    callback = std::move(callback_);
  }
  callback(std::move(status));
}

void
InferenceServiceImpl::handle_async_infer_completion(
    const AsyncInferCompletionContext& context, const Status& job_status,
    const std::vector<torch::Tensor>& outs, LatencyBreakdown breakdown,
    detail::TimingInfo timing_info)
{
  const auto& callback_handle = context.callback_handle;
  if (!callback_handle->TryAcquire()) {
    return;
  }

  if (!job_status.ok()) {
    callback_handle->Invoke(job_status);
    return;
  }

  const auto zero_tp = std::chrono::high_resolution_clock::time_point{};
  if (timing_info.enqueued_time > zero_tp) {
    const auto preprocess_duration = std::chrono::duration<double, std::milli>(
        timing_info.enqueued_time - context.recv_tp);
    breakdown.preprocess_ms = std::max(0.0, preprocess_duration.count());
  } else {
    breakdown.preprocess_ms = 0.0;
  }

  Status populate_status = populate_response(
      context.request, context.reply, outs, context.recv_ms, breakdown,
      context.resolved_model_name);
  if (!populate_status.ok()) {
    callback_handle->Invoke(populate_status);
    return;
  }

  const auto send_tp = std::chrono::high_resolution_clock::now();
  const int64_t send_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              send_tp.time_since_epoch())
                              .count();
  context.reply->set_server_send_ms(send_ms);

  if (timing_info.callback_end_time > zero_tp) {
    const auto postprocess_duration = std::chrono::duration<double, std::milli>(
        send_tp - timing_info.callback_end_time);
    breakdown.postprocess_ms = std::max(0.0, postprocess_duration.count());
  } else {
    breakdown.postprocess_ms = 0.0;
  }

  breakdown.overall_ms = std::max(
      0.0, std::chrono::duration<double, std::milli>(send_tp - context.recv_tp)
               .count());

  context.reply->set_server_preprocess_ms(breakdown.preprocess_ms);
  context.reply->set_server_postprocess_ms(breakdown.postprocess_ms);
  context.reply->set_server_overall_ms(breakdown.overall_ms);

  if (context.metrics && context.metrics->inference_latency != nullptr) {
    const auto latency_ms =
        std::chrono::duration<double, std::milli>(send_tp - context.recv_tp)
            .count();
    context.metrics->inference_latency->Observe(latency_ms);
  }

  callback_handle->Invoke(Status::OK);
}

void
InferenceServiceImpl::HandleModelInferAsync(
    ServerContext* /*context*/, const ModelInferRequest* request,
    ModelInferResponse* reply, std::function<void(Status)> on_done)
{
  auto callback_handle = std::make_shared<CallbackHandle>(std::move(on_done));
  NvtxRange request_scope("grpc_handle_infer_request");

  auto metrics = get_metrics();
  if (metrics && metrics->requests_total != nullptr) {
    metrics->requests_total->Increment();
  }

  auto recv_tp = std::chrono::high_resolution_clock::now();
  record_request_arrival(recv_tp);
  int64_t recv_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        recv_tp.time_since_epoch())
                        .count();

  std::vector<torch::Tensor> inputs;
  std::vector<std::shared_ptr<const void>> input_lifetimes;
  Status status =
      validate_and_convert_inputs(request, inputs, &input_lifetimes);
  if (!status.ok()) {
    callback_handle->Invoke(status);
    return;
  }

  const auto resolved_model_name = resolve_model_name(request->model_name());
  status = submit_job_async(
      inputs,
      [request, reply, recv_tp, recv_ms, metrics, callback_handle,
       resolved_model_name](
          Status const& job_status, const std::vector<torch::Tensor>& outs,
          LatencyBreakdown breakdown, detail::TimingInfo timing_info) mutable {
        handle_async_infer_completion(
            AsyncInferCompletionContext{
                request, reply, callback_handle, metrics, recv_tp, recv_ms,
                resolved_model_name},
            job_status, outs, breakdown, timing_info);
      },
      std::move(input_lifetimes), recv_tp, resolved_model_name);

  if (!status.ok()) {
    callback_handle->Invoke(status);
  }
}

auto
InferenceServiceImpl::populate_response(
    const ModelInferRequest* request, ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs, int64_t recv_ms,
    const LatencyBreakdown& breakdown,
    std::string_view model_name_override) -> Status
{
  if (!model_name_override.empty()) {
    reply->set_model_name(std::string(model_name_override));
  } else {
    reply->set_model_name(request->model_name());
  }
  reply->set_model_version(request->model_version());
  reply->set_server_receive_ms(recv_ms);
  reply->set_server_queue_ms(breakdown.queue_ms);
  reply->set_server_batch_ms(breakdown.batch_ms);
  reply->set_server_submit_ms(breakdown.submit_ms);
  reply->set_server_scheduling_ms(breakdown.scheduling_ms);
  reply->set_server_codelet_ms(breakdown.codelet_ms);
  reply->set_server_inference_ms(breakdown.inference_ms);
  reply->set_server_callback_ms(breakdown.callback_ms);
  reply->set_server_preprocess_ms(breakdown.preprocess_ms);
  reply->set_server_postprocess_ms(breakdown.postprocess_ms);
  reply->set_server_overall_ms(breakdown.overall_ms);
  reply->set_server_total_ms(breakdown.total_ms);
  return fill_output_tensor(reply, outputs);
}

auto
InferenceServiceImpl::ModelInfer(
    ServerContext* context, const ModelInferRequest* request,
    ModelInferResponse* reply) -> Status
{
  std::promise<Status> status_promise;
  auto status_future = status_promise.get_future();
  HandleModelInferAsync(
      context, request, reply, [&status_promise](Status status) {
        status_promise.set_value(std::move(status));
      });
  return status_future.get();
}

namespace {

class AsyncCallDataBase {
 public:
  AsyncCallDataBase() = default;
  AsyncCallDataBase(const AsyncCallDataBase&) = delete;
  auto operator=(const AsyncCallDataBase&) -> AsyncCallDataBase& = delete;
  AsyncCallDataBase(AsyncCallDataBase&&) = default;
  auto operator=(AsyncCallDataBase&&) -> AsyncCallDataBase& = default;
  virtual ~AsyncCallDataBase() = default;
  virtual void Proceed(bool is_ok) = 0;
};

template <typename Request, typename Response>
class UnaryCallData final
    : public AsyncCallDataBase,
      public std::enable_shared_from_this<UnaryCallData<Request, Response>> {
 public:
  using Self = UnaryCallData<Request, Response>;
  using SharedPtr = std::shared_ptr<Self>;
  using RequestMethod = void (inference::GRPCInferenceService::AsyncService::*)(
      grpc::ServerContext*, Request*,
      grpc::ServerAsyncResponseWriter<Response>*, grpc::CompletionQueue*,
      grpc::ServerCompletionQueue*, void*);
  using Handler = std::function<grpc::Status(
      InferenceServiceImpl*, grpc::ServerContext*, const Request*, Response*)>;

  UnaryCallData(
      inference::GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* completion_queue, InferenceServiceImpl* impl,
      RequestMethod request_method, Handler handler)
      : service_(service), cq_(completion_queue), responder_(&ctx_),
        impl_(impl), request_method_(request_method),
        handler_(std::move(handler))
  {
  }

  static void Start(
      inference::GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* completion_queue, InferenceServiceImpl* impl,
      RequestMethod request_method, Handler handler)
  {
    auto call = std::make_shared<Self>(
        service, completion_queue, impl, request_method, std::move(handler));
    call->Proceed(true);
  }

  void Proceed(bool is_ok) override
  {
    switch (status_) {
      case CallStatus::Create: {
        status_ = CallStatus::Process;
        self_ref_ = this->shared_from_this();
        (service_->*request_method_)(
            &ctx_, &request_, &responder_, cq_, cq_, this);
        break;
      }
      case CallStatus::Process: {
        if (!is_ok) {
          status_ = CallStatus::Finish;
          self_ref_.reset();
          return;
        }
        Start(service_, cq_, impl_, request_method_, handler_);
        HandleRequest();
        break;
      }
      case CallStatus::Finish:
        self_ref_.reset();
        break;
    }
  }

 private:
  enum class CallStatus : std::uint8_t { Create, Process, Finish };

  void HandleRequest()
  {
    if (!handler_) {
      status_ = CallStatus::Finish;
      responder_.Finish(
          response_,
          {grpc::StatusCode::INTERNAL, "Unary handler not configured"}, this);
      return;
    }
    auto status = handler_(impl_, &ctx_, &request_, &response_);
    status_ = CallStatus::Finish;
    responder_.Finish(response_, status, this);
  }

  inference::GRPCInferenceService::AsyncService* service_;
  grpc::ServerCompletionQueue* cq_;
  grpc::ServerContext ctx_;
  Request request_;
  Response response_;
  grpc::ServerAsyncResponseWriter<Response> responder_;
  InferenceServiceImpl* impl_;
  RequestMethod request_method_;
  Handler handler_;
  CallStatus status_ = CallStatus::Create;
  SharedPtr self_ref_;
};

class ModelInferCallData final
    : public AsyncCallDataBase,
      public std::enable_shared_from_this<ModelInferCallData> {
 public:
  using Self = ModelInferCallData;
  using SharedPtr = std::shared_ptr<Self>;

  ModelInferCallData(
      inference::GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* completion_queue, InferenceServiceImpl* impl)
      : service_(service), cq_(completion_queue), responder_(&ctx_), impl_(impl)
  {
  }

  static void Start(
      inference::GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* completion_queue, InferenceServiceImpl* impl)
  {
    auto call = std::make_shared<Self>(service, completion_queue, impl);
    call->Proceed(true);
  }

  void Proceed(bool is_ok) override
  {
    using enum CallStatus;
    switch (status_) {
      case Create: {
        status_ = Process;
        self_ref_ = this->shared_from_this();
        service_->RequestModelInfer(
            &ctx_, &request_, &responder_, cq_, cq_, this);
        break;
      }
      case Process: {
        if (!is_ok) {
          status_ = Finish;
          self_ref_.reset();
          return;
        }
        Start(service_, cq_, impl_);
        auto self = this->shared_from_this();
        impl_->HandleModelInferAsync(
            &ctx_, &request_, &response_,
            [self = std::move(self)](const Status& status) {
              self->OnInferenceComplete(status);
            });
        break;
      }
      case Finish:
        self_ref_.reset();
        break;
    }
  }

 private:
  enum class CallStatus : std::uint8_t { Create, Process, Finish };

  void OnInferenceComplete(const Status& status)
  {
    status_ = CallStatus::Finish;
    responder_.Finish(response_, status, this);
  }

  inference::GRPCInferenceService::AsyncService* service_;
  grpc::ServerCompletionQueue* cq_;
  grpc::ServerContext ctx_;
  ModelInferRequest request_;
  ModelInferResponse response_;
  grpc::ServerAsyncResponseWriter<ModelInferResponse> responder_;
  InferenceServiceImpl* impl_;
  CallStatus status_ = CallStatus::Create;
  SharedPtr self_ref_;
};

auto
compute_thread_count() -> std::size_t
{
  return compute_thread_count_from(std::thread::hardware_concurrency());
}

}  // namespace

AsyncServerContext::AsyncServerContext(
    inference::GRPCInferenceService::AsyncService& async_service,
    InferenceServiceImpl& impl)
    : async_service_(&async_service), impl_(&impl)
{
}

void
AsyncServerContext::configure(grpc::ServerBuilder& builder)
{
  builder.RegisterService(async_service_);
  completion_queue_ = builder.AddCompletionQueue();
}

void
AsyncServerContext::start()
{
  if (!completion_queue_ || started_) {
    return;
  }
  started_ = true;
  const std::size_t thread_count = compute_thread_count();
  threads_.reserve(thread_count);
  for (std::size_t i = 0; i < thread_count; ++i) {
    threads_.emplace_back([this]() { this->poll_events(); });
  }

  UnaryCallData<inference::ServerLiveRequest, inference::ServerLiveResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::RequestServerLive,
          std::mem_fn(&InferenceServiceImpl::ServerLive));
  UnaryCallData<inference::ServerReadyRequest, inference::ServerReadyResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::RequestServerReady,
          std::mem_fn(&InferenceServiceImpl::ServerReady));
  UnaryCallData<inference::ModelReadyRequest, inference::ModelReadyResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::RequestModelReady,
          std::mem_fn(&InferenceServiceImpl::ModelReady));
  ModelInferCallData::Start(async_service_, completion_queue_.get(), impl_);
}

void
AsyncServerContext::shutdown()
{
  if (!started_) {
    return;
  }
  started_ = false;
  if (completion_queue_) {
    completion_queue_->Shutdown();
  }
  threads_.clear();
  completion_queue_.reset();
}

auto
AsyncServerContext::started() const -> bool
{
  return started_;
}

auto
AsyncServerContext::thread_count() const -> std::size_t
{
  return threads_.size();
}

void
AsyncServerContext::poll_events()
{
  void* tag = nullptr;
  bool event_ok = false;
  while (completion_queue_ && completion_queue_->Next(&tag, &event_ok)) {
    static_cast<AsyncCallDataBase*>(tag)->Proceed(event_ok);
  }
}

namespace {
auto
configure_server_builder(
    grpc::ServerBuilder& builder, const GrpcServerOptions& options) -> void
{
  builder.AddListeningPort(options.address, grpc::InsecureServerCredentials());
  const int grpc_max_message_bytes =
      options.max_message_bytes >
              static_cast<std::size_t>(std::numeric_limits<int>::max())
          ? std::numeric_limits<int>::max()
          : static_cast<int>(options.max_message_bytes);
  builder.SetMaxReceiveMessageSize(grpc_max_message_bytes);
  builder.SetMaxSendMessageSize(grpc_max_message_bytes);
}

void
run_grpc_server_impl(
    InferenceServiceImpl& service, const GrpcServerOptions& options,
    std::unique_ptr<Server>& server)
{
  inference::GRPCInferenceService::AsyncService async_service;
  AsyncServerContext async_context(async_service, service);

  ServerBuilder builder;
  configure_server_builder(builder, options);
  async_context.configure(builder);

  server = builder.BuildAndStart();
  if (!server) {
    log_error(
        std::format("Failed to start gRPC server on {}", options.address));
    return;
  }
  async_context.start();
  log_info(
      options.verbosity,
      std::format("Server listening on {}", options.address));
  server->Wait();
  async_context.shutdown();
  server.reset();
}
}  // namespace

void
RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::vector<at::ScalarType>& expected_input_types,
    const std::vector<std::vector<int64_t>>& expected_input_dims,
    int max_batch_size, const GrpcServerOptions& options,
    std::unique_ptr<Server>& server)
{
  InferenceServiceImpl service(
      &queue, &reference_outputs, expected_input_types, expected_input_dims,
      max_batch_size, options.default_model_name, options.measured_throughput,
      options.use_cuda, options.measured_throughput_cpu);
  run_grpc_server_impl(service, options, server);
}

void
RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::vector<at::ScalarType>& expected_input_types,
    const GrpcServerOptions& options, std::unique_ptr<Server>& server)
{
  InferenceServiceImpl service(
      &queue, &reference_outputs, expected_input_types,
      options.default_model_name, options.measured_throughput, options.use_cuda,
      options.measured_throughput_cpu);
  run_grpc_server_impl(service, options, server);
}

void
StopServer(Server* server)
{
  if (server != nullptr) {
    server->Shutdown();
  }
}
}  // namespace starpu_server
