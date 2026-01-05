#include "inference_service.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
#include <unordered_map>
#include <unordered_set>
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

namespace {

auto
status_reason(const Status& status) -> std::string
{
  return std::to_string(static_cast<int>(status.error_code()));
}

auto
normalize_names(
    std::vector<std::string> names, std::size_t expected_size,
    std::string_view fallback_prefix,
    std::string_view kind) -> std::vector<std::string>
{
  if (names.empty()) {
    return names;
  }

  const bool any_named = std::any_of(
      names.begin(), names.end(),
      [](const auto& name) { return !name.empty(); });
  if (!any_named) {
    return {};
  }

  if (names.size() != expected_size || expected_size == 0) {
    log_warning(std::format(
        "Configured {} names count ({}) does not match expected count ({}); "
        "ignoring names.",
        kind, names.size(), expected_size));
    return {};
  }

  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i].empty()) {
      names[i] = std::format("{}{}", fallback_prefix, i);
    }
  }

  return names;
}

}  // namespace

namespace {

class AsyncCallDataBase {
 public:
  explicit AsyncCallDataBase() = default;
  AsyncCallDataBase(const AsyncCallDataBase&) = delete;
  auto operator=(const AsyncCallDataBase&) -> AsyncCallDataBase& = delete;
  AsyncCallDataBase(AsyncCallDataBase&&) = default;
  auto operator=(AsyncCallDataBase&&) -> AsyncCallDataBase& = default;
  virtual ~AsyncCallDataBase() = default;
  virtual void Proceed(bool is_ok) = 0;
};

class RpcDoneTag final : public AsyncCallDataBase,
                         public std::enable_shared_from_this<RpcDoneTag> {
 public:
  using OnDone = std::function<void()>;

  static auto Create(OnDone on_done, std::shared_ptr<void> call_guard)
      -> std::shared_ptr<RpcDoneTag>
  {
    return std::shared_ptr<RpcDoneTag>(
        new RpcDoneTag(std::move(on_done), std::move(call_guard)));
  }

  void Arm(grpc::ServerContext* context)
  {
    if (context == nullptr) {
      return;
    }
    self_ref_ = this->shared_from_this();
    context->AsyncNotifyWhenDone(this);
  }

  void Proceed(bool is_ok) override
  {
    if (is_ok && on_done_) {
      on_done_();
    }
    on_done_ = {};
    call_guard_.reset();
    self_ref_.reset();
  }

 private:
  RpcDoneTag(OnDone on_done, std::shared_ptr<void> call_guard)
      : on_done_(std::move(on_done)), call_guard_(std::move(call_guard))
  {
  }

  OnDone on_done_;
  std::shared_ptr<void> call_guard_;
  std::shared_ptr<RpcDoneTag> self_ref_;
};

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)
auto
handle_model_infer_async_test_hooks()
    -> InferenceServiceImpl::HandleModelInferAsyncTestHooks&
{
  static InferenceServiceImpl::HandleModelInferAsyncTestHooks hooks{};
  return hooks;
}
#endif
// GCOVR_EXCL_STOP
}  // namespace


auto
compute_thread_count_from(unsigned concurrency) -> std::size_t
{
  if (concurrency == 0U) {
    return kDefaultGrpcThreads;
  }
  return std::clamp<std::size_t>(concurrency, kMinGrpcThreads, kMaxGrpcThreads);
}

namespace {

using TensorDataByte = std::byte;
using TensorDataPtr = TensorDataByte*;

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

auto
validate_configured_shape(
    const std::vector<int64_t>& shape, const std::vector<int64_t>& expected,
    bool batching_allowed, int max_batch_size) -> Status
{
  const auto rank = static_cast<int64_t>(shape.size());
  const auto expected_rank = static_cast<int64_t>(expected.size());

  auto tails_match = [&](int64_t shape_offset, int64_t expected_offset) {
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
  };

  if (!batching_allowed) {
    if (tails_match(0, 0)) {
      return Status::OK;
    }
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "Input tensor shape does not match configured dimensions or batch "
        "limits"};
  }

  if (rank == 0) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "Input tensor shape does not match configured dimensions or batch "
        "limits"};
  }

  if (tails_match(0, 0)) {
    return Status::OK;
  }

  auto validate_batch_size = [&](int64_t batch_size) -> Status {
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
              "Input tensor shape does not match configured dimensions or "
              "batch limits (batch size {} exceeds configured max of {})",
              batch_size, max_batch_size)};
    }
    return Status::OK;
  };

  if (rank >= 1 && tails_match(1, 1)) {
    const int64_t batch_size = shape.front();
    if (auto status = validate_batch_size(batch_size); !status.ok()) {
      return status;
    }
    return Status::OK;
  }

  if (rank >= 1 && tails_match(1, 0)) {
    const int64_t batch_size = shape.front();
    if (auto status = validate_batch_size(batch_size); !status.ok()) {
      return status;
    }
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "Input tensor shape does not match configured dimensions or batch "
        "limits"};
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

auto
resolve_output_names(
    std::span<const std::string> output_names,
    std::size_t output_count) -> std::vector<std::string>
{
  std::vector<std::string> resolved;
  resolved.reserve(output_count);
  for (std::size_t i = 0; i < output_count; ++i) {
    if (i < output_names.size() && !output_names[i].empty()) {
      resolved.push_back(output_names[i]);
    } else {
      resolved.push_back(std::format("output{}", i));
    }
  }
  return resolved;
}

auto
convert_input_to_tensor(
    const ModelInferRequest::InferInputTensor& input,
    const std::vector<int64_t>& shape, const std::string& raw,
    at::ScalarType dtype, torch::Tensor& tensor,
    std::shared_ptr<const void>* keep_alive) -> Status
{
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

  auto buffer = std::make_shared<std::vector<TensorDataByte>>(raw.size());
  if (!raw.empty()) {
    std::memcpy(buffer->data(), raw.data(), raw.size());
  }
  auto deleter = [buffer](void* /*unused*/) mutable { buffer.reset(); };

  TensorDataPtr tensor_data = buffer->data();
  tensor = torch::from_blob(tensor_data, shape, deleter, options);
  if (keep_alive != nullptr) {
    *keep_alive = std::shared_ptr<const void>(buffer, buffer->data());
  }
  return Status::OK;
}

auto
fill_output_tensor(
    ModelInferResponse* reply, const std::vector<torch::Tensor>& outputs,
    std::span<const std::size_t> output_indices,
    std::span<const std::string> output_names) -> Status
{
  for (std::size_t pos = 0; pos < output_indices.size(); ++pos) {
    const std::size_t output_index = output_indices[pos];
    if (output_index >= outputs.size()) {
      return {
          grpc::StatusCode::INVALID_ARGUMENT,
          "Requested output index out of range"};
    }
    const auto& original = outputs[output_index];
    torch::Tensor out = original;
    if (!original.device().is_cpu()) {
      out = original.to(torch::kCPU);
    }
    auto* out_tensor = reply->add_outputs();
    if (output_index < output_names.size()) {
      out_tensor->set_name(output_names[output_index]);
    } else {
      out_tensor->set_name(std::format("output{}", output_index));
    }
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
    std::string default_model_name,
    std::vector<std::string> expected_input_names,
    std::vector<std::string> expected_output_names)
    : queue_(queue), reference_outputs_(reference_outputs),
      expected_input_types_(std::move(expected_input_types)),
      expected_input_dims_(std::move(expected_input_dims)),
      max_batch_size_(max_batch_size),
      default_model_name_(std::move(default_model_name))
{
  expected_input_names_ = normalize_names(
      std::move(expected_input_names), expected_input_types_.size(), "input",
      "input");
  const std::size_t output_count =
      reference_outputs_ != nullptr ? reference_outputs_->size() : 0U;
  expected_output_names_ = normalize_names(
      std::move(expected_output_names), output_count, "output", "output");
}

InferenceServiceImpl::InferenceServiceImpl(
    InferenceQueue* queue, const std::vector<torch::Tensor>* reference_outputs,
    std::vector<at::ScalarType> expected_input_types,
    std::string default_model_name,
    std::vector<std::string> expected_input_names,
    std::vector<std::string> expected_output_names)
    : InferenceServiceImpl(
          queue, reference_outputs, std::move(expected_input_types), {}, 0,
          std::move(default_model_name), std::move(expected_input_names),
          std::move(expected_output_names))
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
  const bool ready = queue_ != nullptr && !queue_->is_shutdown();
  reply->set_ready(ready);
  return Status::OK;
}

auto
InferenceServiceImpl::ModelReady(
    ServerContext* /*context*/, const ModelReadyRequest* request,
    ModelReadyResponse* reply) -> Status
{
  bool ready = queue_ != nullptr && !queue_->is_shutdown();
  if (ready && !default_model_name_.empty()) {
    const auto& requested_name = request->name();
    if (!requested_name.empty() && requested_name != default_model_name_) {
      ready = false;
    }
  }
  reply->set_ready(ready);
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

  bool any_named = false;
  bool any_unnamed = false;
  for (const auto& input : request->inputs()) {
    if (input.name().empty()) {
      any_unnamed = true;
    } else {
      any_named = true;
    }
  }
  if (any_named && any_unnamed) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "All input tensors must include a name when using named inputs"};
  }

  const bool has_expected_names = !expected_input_names_.empty();
  if (has_expected_names && !any_named) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "Input tensor names must be provided"};
  }
  const bool use_name_mapping = has_expected_names;

  std::unordered_map<std::string_view, std::size_t> expected_index_by_name;
  if (use_name_mapping) {
    expected_index_by_name.reserve(expected_input_names_.size());
    for (std::size_t i = 0; i < expected_input_names_.size(); ++i) {
      const auto& name = expected_input_names_[i];
      if (name.empty()) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format("Configured input name missing at index {}", i)};
      }
      const auto [it, inserted] = expected_index_by_name.emplace(name, i);
      if (!inserted) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format("Configured input name '{}' is duplicated", name)};
      }
    }
  }

  const std::size_t expected_count = expected_input_types_.size();
  std::vector<torch::Tensor> ordered_inputs(expected_count);
  std::vector<std::shared_ptr<const void>> ordered_lifetimes;
  if (input_lifetimes != nullptr) {
    ordered_lifetimes.resize(expected_count);
  }
  std::vector<bool> filled(expected_count, false);

  for (int i = 0; i < request->inputs_size(); ++i) {
    const auto& input = request->inputs(i);
    const auto& raw = request->raw_input_contents(i);
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

    std::size_t expected_index = static_cast<std::size_t>(i);
    if (use_name_mapping) {
      const auto it = expected_index_by_name.find(input.name());
      if (it == expected_index_by_name.end()) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format("Unexpected input tensor name '{}'", input.name())};
      }
      expected_index = it->second;
      if (filled[expected_index]) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format("Input tensor name '{}' is duplicated", input.name())};
      }
    }

    at::ScalarType dtype = at::kFloat;
    Status status =
        parse_input_dtype(input, expected_input_types_[expected_index], dtype);
    if (!status.ok()) {
      return status;
    }

    torch::Tensor tensor;
    std::shared_ptr<const void> tensor_guard;
    status = convert_input_to_tensor(
        input, shape, raw, dtype, tensor,
        input_lifetimes != nullptr ? &tensor_guard : nullptr);
    if (!status.ok()) {
      return status;
    }

    if (expected_index < expected_input_dims_.size()) {
      const auto& expected_dims = expected_input_dims_[expected_index];
      const bool batching_allowed = (max_batch_size_ > 0);
      status = validate_configured_shape(
          shape, expected_dims, batching_allowed, max_batch_size_);
      if (!status.ok()) {
        return status;
      }
    }

    ordered_inputs[expected_index] = std::move(tensor);
    if (input_lifetimes != nullptr) {
      ordered_lifetimes[expected_index] = std::move(tensor_guard);
    }
    filled[expected_index] = true;
  }

  if (use_name_mapping) {
    for (std::size_t i = 0; i < filled.size(); ++i) {
      if (!filled[i]) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format("Missing input tensor '{}'", expected_input_names_[i])};
      }
    }
  }

  inputs = std::move(ordered_inputs);
  if (input_lifetimes != nullptr) {
    *input_lifetimes = std::move(ordered_lifetimes);
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

auto
InferenceServiceImpl::next_request_id() -> int
{
  constexpr int kMaxRequestId = std::numeric_limits<int>::max();
  int current = next_request_id_.load(std::memory_order_relaxed);
  while (true) {
    int issued = 0;
    int next = 1;
    if (current >= 0 && current < kMaxRequestId) {
      issued = current;
      next = current + 1;
    }
    if (next_request_id_.compare_exchange_weak(
            current, next, std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
      return issued;
    }
  }
}

auto
InferenceServiceImpl::submit_job_async(
    const std::vector<torch::Tensor>& inputs, AsyncJobCallback on_complete,
    std::vector<std::shared_ptr<const void>> input_lifetimes,
    std::shared_ptr<std::atomic<bool>> cancel_flag,
    MonotonicClock::time_point receive_time, std::string model_name) -> Status
{
  auto resolved_model_name = resolve_model_name(std::move(model_name));
  auto job = client_utils::create_job(
      inputs, *reference_outputs_, next_request_id(),
      std::move(input_lifetimes), receive_time, std::move(resolved_model_name));
  job->set_cancelled_flag(std::move(cancel_flag));

  NvtxRange submit_scope("grpc_submit_starpu");

  job->set_on_complete([job, callback = std::move(on_complete)](
                           std::vector<torch::Tensor> outs,
                           double latency_ms) mutable {
    const auto& info = job->timing_info();
    auto timing = build_latency_breakdown(info, latency_ms);
    detail::TimingInfo copied_info = info;

    if (outs.empty()) {
      std::optional<InferenceServiceImpl::AsyncFailureInfo> failure_info;
      Status status = Status::OK;
      if (auto job_failure = job->take_failure_info()) {
        const std::string reason = job_failure->reason;
        const std::string detail_message = job_failure->message;
        std::string message;
        if (!reason.empty() && !detail_message.empty()) {
          message =
              std::format("Inference failed ({}): {}", reason, detail_message);
        } else if (!reason.empty()) {
          message = std::format("Inference failed ({})", reason);
        } else if (!detail_message.empty()) {
          message = std::format("Inference failed: {}", detail_message);
        } else {
          message = "Inference failed";
        }
        status = {grpc::StatusCode::INTERNAL, message};
        AsyncFailureInfo info{};
        info.stage = std::move(job_failure->stage);
        info.reason = std::move(job_failure->reason);
        info.metrics_reported = job_failure->metrics_reported;
        failure_info = std::move(info);
      } else {
        AsyncFailureInfo info{};
        info.stage = "execution";
        info.reason = "empty_output";
        failure_info = std::move(info);
        status = {grpc::StatusCode::INTERNAL, "Inference failed"};
      }
      callback(status, {}, timing, copied_info, std::move(failure_info));
      return;
    }

    callback(Status::OK, std::move(outs), timing, copied_info, std::nullopt);
  });

  bool pushed = false;
  bool queue_full = false;
  {
    NvtxRange queue_scope("grpc_submit_starpu_queue");
    pushed = queue_->push(job, &queue_full);
    if (pushed) {
      const auto enqueued_now = MonotonicClock::now();
      job->timing_info().enqueued_time = enqueued_now;
      job->timing_info().last_enqueued_time = enqueued_now;
    }
  }
  if (!pushed) {
    if (queue_full) {
      increment_inference_failure("enqueue", "queue_full", job->model_name());
      increment_rejected_requests();
      BatchingTraceLogger::instance().log_request_rejected(queue_->size());
      return {grpc::StatusCode::RESOURCE_EXHAUSTED, "Inference queue is full"};
    }
    increment_inference_failure(
        "enqueue", "queue_unavailable", job->model_name());
    return {grpc::StatusCode::UNAVAILABLE, "Inference queue unavailable"};
  }
  if (auto& tracer = BatchingTraceLogger::instance(); tracer.enabled()) {
    tracer.log_request_enqueued(
        job->get_request_id(), job->model_name(), /*is_warmup=*/false,
        job->timing_info().last_enqueued_time);
  }
  return Status::OK;
}

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)
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

  const auto receive_time = MonotonicClock::now();
  if (Status submit_status = submit_job_async(
          inputs,
          [result_promise](
              Status status, std::vector<torch::Tensor> outs,
              const LatencyBreakdown& cb_breakdown,
              const detail::TimingInfo& cb_timing_info,
              std::optional<AsyncFailureInfo> /*failure_info*/) {
            result_promise->set_value(JobResult{
                std::move(status), std::move(outs), cb_breakdown,
                cb_timing_info});
          },
          std::move(input_lifetimes), std::shared_ptr<std::atomic<bool>>{},
          receive_time);
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

void
InferenceServiceImpl::TestAccessor::SetHandleModelInferAsyncTestHooks(
    HandleModelInferAsyncTestHooks hooks)
{
  handle_model_infer_async_test_hooks() = std::move(hooks);
}

void
InferenceServiceImpl::TestAccessor::ClearHandleModelInferAsyncTestHooks()
{
  handle_model_infer_async_test_hooks() = HandleModelInferAsyncTestHooks{};
}

auto
InferenceServiceImpl::TestAccessor::NormalizeNamesForTest(
    std::vector<std::string> names, std::size_t expected_size,
    std::string_view fallback_prefix,
    std::string_view kind) -> std::vector<std::string>
{
  return normalize_names(
      std::move(names), expected_size, fallback_prefix, kind);
}
#endif
// GCOVR_EXCL_STOP

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

auto
InferenceServiceImpl::CallbackHandle::Invoke(Status status) -> bool
{
  std::function<void(Status)> callback;
  {
    std::scoped_lock lock(mutex_);
    if (!callback_) {
      return false;
    }
    consumed_ = true;
    callback = std::move(callback_);
  }
  callback(std::move(status));
  return true;
}

void
InferenceServiceImpl::handle_async_infer_completion(
    const AsyncInferCompletionContext& context, const Status& job_status,
    const std::vector<torch::Tensor>& outs, LatencyBreakdown breakdown,
    detail::TimingInfo timing_info)
{
  const auto& callback_handle = context.callback_handle;
  if (context.cancel_flag != nullptr &&
      context.cancel_flag->load(std::memory_order_acquire)) {
    return;
  }
  if (!callback_handle->TryAcquire()) {
    return;
  }
  if (context.cancel_flag != nullptr &&
      context.cancel_flag->load(std::memory_order_acquire)) {
    return;
  }

  if (!job_status.ok()) {
    increment_request_status(
        static_cast<int>(job_status.error_code()), context.resolved_model_name);
    const bool metrics_reported =
        context.failure_info && context.failure_info->metrics_reported;
    if (!metrics_reported) {
      std::string stage = "execution";
      std::string reason = status_reason(job_status);
      if (context.failure_info) {
        if (!context.failure_info->stage.empty()) {
          stage = context.failure_info->stage;
        }
        if (!context.failure_info->reason.empty()) {
          reason = context.failure_info->reason;
        }
      }
      increment_inference_failure(stage, reason, context.resolved_model_name);
    }
    callback_handle->Invoke(job_status);
    return;
  }

  const auto zero_tp = MonotonicClock::time_point{};
  if (timing_info.enqueued_time > zero_tp) {
    const auto preprocess_duration = std::chrono::duration<double, std::milli>(
        timing_info.enqueued_time - context.recv_tp);
    breakdown.preprocess_ms = std::max(0.0, preprocess_duration.count());
  } else {
    breakdown.preprocess_ms = 0.0;
  }

  const auto output_names =
      context.output_names != nullptr
          ? std::span<const std::string>(*context.output_names)
          : std::span<const std::string>{};
  Status populate_status = populate_response(
      context.request, context.reply, outs, context.recv_ms, breakdown,
      context.resolved_model_name, /*set_prepost_overall=*/false, output_names);
  if (!populate_status.ok()) {
    increment_request_status(
        static_cast<int>(populate_status.error_code()),
        context.resolved_model_name);
    increment_inference_failure(
        "postprocess", status_reason(populate_status),
        context.resolved_model_name);
    callback_handle->Invoke(populate_status);
    return;
  }

  const auto send_tp = MonotonicClock::now();
  const int64_t send_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
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

  if (context.metrics && context.metrics->inference_latency() != nullptr) {
    const auto latency_ms =
        std::chrono::duration<double, std::milli>(send_tp - context.recv_tp)
            .count();
    context.metrics->inference_latency()->Observe(latency_ms);
  }

  observe_latency_breakdown(
      breakdown.queue_ms, breakdown.batch_ms, breakdown.submit_ms,
      breakdown.scheduling_ms, breakdown.codelet_ms, breakdown.inference_ms,
      breakdown.callback_ms, breakdown.preprocess_ms, breakdown.postprocess_ms);

  if (context.cancel_flag != nullptr &&
      context.cancel_flag->load(std::memory_order_acquire)) {
    return;
  }

  increment_request_status(
      static_cast<int>(grpc::StatusCode::OK), context.resolved_model_name);
  callback_handle->Invoke(Status::OK);
}

void
InferenceServiceImpl::HandleModelInferAsync(
    ServerContext* context, const ModelInferRequest* request,
    ModelInferResponse* reply, std::function<void(Status)> on_done,
    std::shared_ptr<void> call_guard)
{
  auto callback_handle = std::make_shared<CallbackHandle>(std::move(on_done));
  auto cancel_flag = std::make_shared<std::atomic<bool>>(false);
#if defined(STARPU_TESTING)
  auto& test_hooks = handle_model_infer_async_test_hooks();
  if (test_hooks.on_cancel_flag_created) {
    test_hooks.on_cancel_flag_created(cancel_flag);
  }
#endif
  NvtxRange request_scope("grpc_handle_infer_request");

  auto metrics = get_metrics();
  if (metrics && metrics->requests_total() != nullptr) {
    metrics->requests_total()->Increment();
  }

  const auto resolved_model_name = resolve_model_name(request->model_name());
  if (context != nullptr && call_guard) {
    auto is_context_cancelled = [context
#if defined(STARPU_TESTING)
                                 ,
                                 cancel_override =
                                     test_hooks.is_cancelled_override
#endif
    ]() -> bool {
      if (context == nullptr) {
        return false;
      }
#if defined(STARPU_TESTING)
      if (cancel_override) {
        if (auto override = cancel_override(context); override.has_value()) {
          return *override;
        }
      }
#endif
      return context->IsCancelled();
    };
    auto on_cancel = [context, cancel_flag, callback_handle,
                      resolved_model_name, is_context_cancelled]() {
      if (context == nullptr || !is_context_cancelled()) {
        return;
      }
      if (cancel_flag->exchange(true, std::memory_order_acq_rel)) {
        return;
      }
      const bool invoked = callback_handle->Invoke(
          {grpc::StatusCode::CANCELLED, "Request cancelled"});
      if (invoked) {
        increment_request_status(
            static_cast<int>(grpc::StatusCode::CANCELLED), resolved_model_name);
        increment_inference_failure(
            "cancel", "client_cancelled", resolved_model_name);
      }
    };
#if defined(STARPU_TESTING)
    if (test_hooks.on_cancel_ready) {
      test_hooks.on_cancel_ready(on_cancel);
    }
#endif
    auto done_tag = RpcDoneTag::Create(on_cancel, std::move(call_guard));
    done_tag->Arm(context);
    if (is_context_cancelled()) {
      on_cancel();
      return;
    }
  }
  auto recv_tp = MonotonicClock::now();
  int64_t recv_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();

  std::vector<torch::Tensor> inputs;
  std::vector<std::shared_ptr<const void>> input_lifetimes;
  Status status =
      validate_and_convert_inputs(request, inputs, &input_lifetimes);
  if (!status.ok()) {
    if (cancel_flag->load(std::memory_order_acquire)) {
      return;
    }
    increment_request_status(
        static_cast<int>(status.error_code()), resolved_model_name);
    increment_inference_failure(
        "preprocess", status_reason(status), resolved_model_name);
    callback_handle->Invoke(status);
    return;
  }

  if (cancel_flag->load(std::memory_order_acquire)) {
    return;
  }

  const auto* output_names = &expected_output_names_;
  status = submit_job_async(
      inputs,
      [request, reply, recv_tp, recv_ms, metrics, callback_handle,
       resolved_model_name, cancel_flag, output_names](
          Status const& job_status, const std::vector<torch::Tensor>& outs,
          LatencyBreakdown breakdown, detail::TimingInfo timing_info,
          std::optional<AsyncFailureInfo> failure_info) mutable {
        handle_async_infer_completion(
            AsyncInferCompletionContext{
                request, reply, callback_handle, metrics, recv_tp, recv_ms,
                resolved_model_name, output_names, cancel_flag,
                std::move(failure_info)},
            job_status, outs, breakdown, timing_info);
      },
      std::move(input_lifetimes), cancel_flag, recv_tp, resolved_model_name);

  if (!status.ok()) {
    if (cancel_flag->load(std::memory_order_acquire)) {
      return;
    }
    increment_request_status(
        static_cast<int>(status.error_code()), resolved_model_name);
    callback_handle->Invoke(status);
  }
}

auto
InferenceServiceImpl::populate_response(
    const ModelInferRequest* request, ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs, int64_t recv_ms,
    const LatencyBreakdown& breakdown, std::string_view model_name_override,
    bool set_prepost_overall,
    std::span<const std::string> output_names) -> Status
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
  if (set_prepost_overall) {
    reply->set_server_preprocess_ms(breakdown.preprocess_ms);
    reply->set_server_postprocess_ms(breakdown.postprocess_ms);
    reply->set_server_overall_ms(breakdown.overall_ms);
  }
  reply->set_server_total_ms(breakdown.total_ms);

  const auto resolved_names =
      resolve_output_names(output_names, outputs.size());
  std::vector<std::size_t> output_indices;
  if (request->outputs_size() > 0) {
    std::unordered_map<std::string_view, std::size_t> index_by_name;
    index_by_name.reserve(resolved_names.size());
    for (std::size_t i = 0; i < resolved_names.size(); ++i) {
      const auto [it, inserted] = index_by_name.emplace(resolved_names[i], i);
      if (!inserted) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format(
                "Configured output name '{}' is duplicated",
                resolved_names[i])};
      }
    }

    std::unordered_set<std::string_view> seen;
    output_indices.reserve(request->outputs_size());
    for (const auto& requested : request->outputs()) {
      if (requested.name().empty()) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            "Requested output name must be non-empty"};
      }
      const auto it = index_by_name.find(requested.name());
      if (it == index_by_name.end()) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format(
                "Requested output '{}' is not available", requested.name())};
      }
      if (!seen.insert(it->first).second) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format(
                "Requested output '{}' is duplicated", requested.name())};
      }
      output_indices.push_back(it->second);
    }
  } else {
    output_indices.reserve(outputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      output_indices.push_back(i);
    }
  }

  return fill_output_tensor(reply, outputs, output_indices, resolved_names);
}

#if defined(STARPU_TESTING)
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
#endif

namespace {

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
        auto call_guard = self;
        impl_->HandleModelInferAsync(
            &ctx_, &request_, &response_,
            [self = std::move(self)](const Status& status) {
              self->OnInferenceComplete(status);
            },
            std::move(call_guard));
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

void
run_grpc_server_impl(
    InferenceServiceImpl& service, const GrpcServerOptions& options,
    std::unique_ptr<Server>& server)
{
  inference::GRPCInferenceService::AsyncService async_service;
  AsyncServerContext async_context(async_service, service);

  ServerBuilder builder;
  builder.AddListeningPort(options.address, grpc::InsecureServerCredentials());
  async_context.configure(builder);
  const int grpc_max_message_bytes =
      options.max_message_bytes >
              static_cast<std::size_t>(std::numeric_limits<int>::max())
          ? std::numeric_limits<int>::max()
          : static_cast<int>(options.max_message_bytes);
  builder.SetMaxReceiveMessageSize(grpc_max_message_bytes);
  builder.SetMaxSendMessageSize(grpc_max_message_bytes);

  server = builder.BuildAndStart();
  if (!server) {
    log_error(
        std::format("Failed to start gRPC server on {}", options.address));
    set_server_health(false);
    return;
  }
  set_server_health(true);
  async_context.start();
  log_info(
      options.verbosity,
      std::format("Server listening on {}", options.address));
  server->Wait();
  set_server_health(false);
  async_context.shutdown();
  server.reset();
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

void
AsyncServerContext::poll_events()
{
  void* tag = nullptr;
  bool event_ok = false;
  while (completion_queue_ && completion_queue_->Next(&tag, &event_ok)) {
    static_cast<AsyncCallDataBase*>(tag)->Proceed(event_ok);
  }
}

void
RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::vector<at::ScalarType>& expected_input_types,
    const std::vector<std::vector<int64_t>>& expected_input_dims,
    const std::vector<std::string>& expected_input_names,
    const std::vector<std::string>& expected_output_names, int max_batch_size,
    const GrpcServerOptions& options, std::unique_ptr<Server>& server)
{
  InferenceServiceImpl service(
      &queue, &reference_outputs, expected_input_types, expected_input_dims,
      max_batch_size, options.default_model_name,
      std::vector<std::string>(
          expected_input_names.begin(), expected_input_names.end()),
      std::vector<std::string>(
          expected_output_names.begin(), expected_output_names.end()));
  run_grpc_server_impl(service, options, server);
}

void
RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::vector<at::ScalarType>& expected_input_types,
    const std::vector<std::string>& expected_input_names,
    const std::vector<std::string>& expected_output_names,
    const GrpcServerOptions& options, std::unique_ptr<Server>& server)
{
  InferenceServiceImpl service(
      &queue, &reference_outputs,
      std::vector<at::ScalarType>(
          expected_input_types.begin(), expected_input_types.end()),
      options.default_model_name,
      std::vector<std::string>(
          expected_input_names.begin(), expected_input_names.end()),
      std::vector<std::string>(
          expected_output_names.begin(), expected_output_names.end()));
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
