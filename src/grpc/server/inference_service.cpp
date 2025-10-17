#include "inference_service.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <format>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "monitoring/metrics.hpp"
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

using TensorDataPtr = void*;

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
    return Status::OK;
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

  auto alias = std::shared_ptr<const void>(request_guard, raw.data());
  auto holder = std::const_pointer_cast<void>(alias);
  auto deleter = [holder](TensorDataPtr /*unused*/) mutable {
    auto keep = holder;
    keep.reset();
  };

  tensor = torch::from_blob(holder.get(), shape, deleter, options);
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
    std::vector<std::vector<int64_t>> expected_input_dims, int max_batch_size)
    : queue_(queue), reference_outputs_(reference_outputs),
      expected_input_types_(std::move(expected_input_types)),
      expected_input_dims_(std::move(expected_input_dims)),
      max_batch_size_(max_batch_size)
{
}

InferenceServiceImpl::InferenceServiceImpl(
    InferenceQueue* queue, const std::vector<torch::Tensor>* reference_outputs,
    std::vector<at::ScalarType> expected_input_types)
    : queue_(queue), reference_outputs_(reference_outputs),
      expected_input_types_(std::move(expected_input_types))
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
  using duration_f = std::chrono::duration<double, std::milli>;
  LatencyBreakdown timing{};
  timing.queue_ms = duration_f(info.dequeued_time - info.enqueued_time).count();
  timing.batch_ms = std::max(
      0.0,
      duration_f(info.batch_collect_end_time - info.batch_collect_start_time)
          .count());
  timing.submit_ms = std::max(
      0.0, duration_f(
               info.before_starpu_submitted_time - info.batch_collect_end_time)
               .count());
  timing.scheduling_ms =
      duration_f(info.codelet_start_time - info.before_starpu_submitted_time)
          .count();
  timing.codelet_ms =
      duration_f(info.codelet_end_time - info.codelet_start_time).count();
  timing.inference_ms =
      duration_f(info.callback_start_time - info.inference_start_time).count();
  timing.callback_ms =
      duration_f(info.callback_end_time - info.callback_start_time).count();
  timing.total_ms = latency_ms;
  return timing;
}

auto
InferenceServiceImpl::submit_job_async(
    const std::vector<torch::Tensor>& inputs, AsyncJobCallback on_complete,
    std::vector<std::shared_ptr<const void>> input_lifetimes,
    std::chrono::high_resolution_clock::time_point receive_time) -> Status
{
  auto job = client_utils::create_job(
      inputs, *reference_outputs_, next_request_id_++,
      std::move(input_lifetimes), receive_time);

  NvtxRange submit_scope("grpc_submit_starpu");

  job->set_on_complete(
      [job_ptr = job, on_complete = std::move(on_complete)](
          const std::vector<torch::Tensor>& outs, double latency_ms) mutable {
        const auto& info = job_ptr->timing_info();
        auto timing = build_latency_breakdown(info, latency_ms);
        detail::TimingInfo copied_info = info;
        if (outs.empty()) {
          on_complete(
              {grpc::StatusCode::INTERNAL, "Inference failed"}, {}, timing,
              copied_info);
          return;
        }

        auto outputs_copy = outs;
        on_complete(Status::OK, std::move(outputs_copy), timing, copied_info);
      });

  bool pushed = false;
  {
    NvtxRange queue_scope("grpc_submit_starpu_queue");
    pushed = queue_->push(job);
  }
  if (!pushed) {
    return {grpc::StatusCode::UNAVAILABLE, "Inference queue unavailable"};
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
              LatencyBreakdown cb_breakdown,
              const detail::TimingInfo& cb_timing_info) {
            result_promise->set_value(JobResult{
                std::move(status), std::move(outs), std::move(cb_breakdown),
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
  breakdown = std::move(result.breakdown);
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
    const ModelInferRequest* request, ModelInferResponse* reply,
    const std::shared_ptr<CallbackHandle>& callback_handle,
    std::shared_ptr<MetricsRegistry> metrics,
    std::chrono::high_resolution_clock::time_point recv_tp, int64_t recv_ms,
    const Status& job_status, const std::vector<torch::Tensor>& outs,
    LatencyBreakdown breakdown, detail::TimingInfo timing_info)
{
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
        timing_info.enqueued_time - recv_tp);
    breakdown.preprocess_ms = std::max(0.0, preprocess_duration.count());
  } else {
    breakdown.preprocess_ms = 0.0;
  }

  Status populate_status =
      populate_response(request, reply, outs, recv_ms, breakdown);
  if (!populate_status.ok()) {
    callback_handle->Invoke(populate_status);
    return;
  }

  const auto send_tp = std::chrono::high_resolution_clock::now();
  const int64_t send_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              send_tp.time_since_epoch())
                              .count();
  reply->set_server_send_ms(send_ms);

  if (timing_info.callback_end_time > zero_tp) {
    const auto postprocess_duration = std::chrono::duration<double, std::milli>(
        send_tp - timing_info.callback_end_time);
    breakdown.postprocess_ms = std::max(0.0, postprocess_duration.count());
  } else {
    breakdown.postprocess_ms = 0.0;
  }

  breakdown.overall_ms = std::max(
      0.0,
      std::chrono::duration<double, std::milli>(send_tp - recv_tp).count());

  reply->set_server_preprocess_ms(breakdown.preprocess_ms);
  reply->set_server_postprocess_ms(breakdown.postprocess_ms);
  reply->set_server_overall_ms(breakdown.overall_ms);

  if (metrics && metrics->inference_latency != nullptr) {
    const auto latency_ms =
        std::chrono::duration<double, std::milli>(send_tp - recv_tp).count();
    metrics->inference_latency->Observe(latency_ms);
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

  status = submit_job_async(
      inputs,
      [this, request, reply, recv_tp, recv_ms, metrics, callback_handle](
          Status const& job_status, const std::vector<torch::Tensor>& outs,
          LatencyBreakdown breakdown, detail::TimingInfo timing_info) mutable {
        handle_async_infer_completion(
            request, reply, callback_handle, metrics, recv_tp, recv_ms,
            job_status, outs, std::move(breakdown), timing_info);
      },
      std::move(input_lifetimes), recv_tp);

  if (!status.ok()) {
    callback_handle->Invoke(status);
  }
}

auto
InferenceServiceImpl::populate_response(
    const ModelInferRequest* request, ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs, int64_t recv_ms,
    const LatencyBreakdown& breakdown) -> Status
{
  reply->set_model_name(request->model_name());
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
  explicit AsyncCallDataBase() = default;
  AsyncCallDataBase(const AsyncCallDataBase&) = delete;
  auto operator=(const AsyncCallDataBase&) -> AsyncCallDataBase& = delete;
  AsyncCallDataBase(AsyncCallDataBase&&) = default;
  auto operator=(AsyncCallDataBase&&) -> AsyncCallDataBase& = default;
  virtual ~AsyncCallDataBase() = default;
  virtual void Proceed(bool is_ok) = 0;
};

template <typename Request, typename Response>
class UnaryCallData final : public AsyncCallDataBase {
 public:
  using RequestMethod = void (inference::GRPCInferenceService::AsyncService::*)(
      grpc::ServerContext*, Request*,
      grpc::ServerAsyncResponseWriter<Response>*, grpc::CompletionQueue*,
      grpc::ServerCompletionQueue*, void*);
  using HandlerMethod = grpc::Status (InferenceServiceImpl::*)(
      grpc::ServerContext*, const Request*, Response*);

  UnaryCallData(
      inference::GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* completion_queue, InferenceServiceImpl* impl,
      RequestMethod request_method, HandlerMethod handler)
      : service_(service), cq_(completion_queue), responder_(&ctx_),
        impl_(impl), request_method_(request_method), handler_(handler)
  {
    Proceed(true);
  }

  void Proceed(bool is_ok) override
  {
    switch (status_) {
      case CallStatus::Create:
        status_ = CallStatus::Process;
        (service_->*request_method_)(
            &ctx_, &request_, &responder_, cq_, cq_, this);
        break;
      case CallStatus::Process:
        if (!is_ok) {
          status_ = CallStatus::Finish;
          delete this;
          return;
        }
        new UnaryCallData(service_, cq_, impl_, request_method_, handler_);
        HandleRequest();
        break;
      case CallStatus::Finish:
        delete this;
        break;
    }
  }

 private:
  enum class CallStatus : std::uint8_t { Create, Process, Finish };

  void HandleRequest()
  {
    auto status = (impl_->*handler_)(&ctx_, &request_, &response_);
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
  HandlerMethod handler_;
  CallStatus status_ = CallStatus::Create;
};

class ModelInferCallData final : public AsyncCallDataBase {
 public:
  ModelInferCallData(
      inference::GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* completion_queue, InferenceServiceImpl* impl)
      : service_(service), cq_(completion_queue), responder_(&ctx_), impl_(impl)
  {
    Proceed(true);
  }

  void Proceed(bool is_ok) override
  {
    switch (status_) {
      case CallStatus::Create:
        status_ = CallStatus::Process;
        service_->RequestModelInfer(
            &ctx_, &request_, &responder_, cq_, cq_, this);
        break;
      case CallStatus::Process:
        if (!is_ok) {
          status_ = CallStatus::Finish;
          delete this;
          return;
        }
        new ModelInferCallData(service_, cq_, impl_);
        impl_->HandleModelInferAsync(
            &ctx_, &request_, &response_, [this](const Status& status) {
              status_ = CallStatus::Finish;
              responder_.Finish(response_, status, this);
            });
        break;
      case CallStatus::Finish:
        delete this;
        break;
    }
  }

 private:
  enum class CallStatus : std::uint8_t { Create, Process, Finish };

  inference::GRPCInferenceService::AsyncService* service_;
  grpc::ServerCompletionQueue* cq_;
  grpc::ServerContext ctx_;
  ModelInferRequest request_;
  ModelInferResponse response_;
  grpc::ServerAsyncResponseWriter<ModelInferResponse> responder_;
  InferenceServiceImpl* impl_;
  CallStatus status_ = CallStatus::Create;
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

  new UnaryCallData<
      inference::ServerLiveRequest, inference::ServerLiveResponse>(
      async_service_, completion_queue_.get(), impl_,
      &inference::GRPCInferenceService::AsyncService::RequestServerLive,
      &InferenceServiceImpl::ServerLive);
  new UnaryCallData<
      inference::ServerReadyRequest, inference::ServerReadyResponse>(
      async_service_, completion_queue_.get(), impl_,
      &inference::GRPCInferenceService::AsyncService::RequestServerReady,
      &InferenceServiceImpl::ServerReady);
  new UnaryCallData<
      inference::ModelReadyRequest, inference::ModelReadyResponse>(
      async_service_, completion_queue_.get(), impl_,
      &inference::GRPCInferenceService::AsyncService::RequestModelReady,
      &InferenceServiceImpl::ModelReady);
  new ModelInferCallData(async_service_, completion_queue_.get(), impl_);
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

void
RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::vector<at::ScalarType>& expected_input_types,
    const std::vector<std::vector<int64_t>>& expected_input_dims,
    int max_batch_size, const std::string& address,
    std::size_t max_message_bytes, VerbosityLevel verbosity,
    std::unique_ptr<Server>& server)
{
  InferenceServiceImpl service(
      &queue, &reference_outputs, expected_input_types, expected_input_dims,
      max_batch_size);

  inference::GRPCInferenceService::AsyncService async_service;
  AsyncServerContext async_context(async_service, service);

  ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  async_context.configure(builder);
  const int grpc_max_message_bytes =
      max_message_bytes >
              static_cast<std::size_t>(std::numeric_limits<int>::max())
          ? std::numeric_limits<int>::max()
          : static_cast<int>(max_message_bytes);
  builder.SetMaxReceiveMessageSize(grpc_max_message_bytes);
  builder.SetMaxSendMessageSize(grpc_max_message_bytes);

  server = builder.BuildAndStart();
  if (!server) {
    log_error(std::format("Failed to start gRPC server on {}", address));
    return;
  }
  async_context.start();
  log_info(verbosity, std::format("Server listening on {}", address));
  server->Wait();
  async_context.shutdown();
  server.reset();
}

void
RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::vector<at::ScalarType>& expected_input_types,
    const std::string& address, std::size_t max_message_bytes,
    VerbosityLevel verbosity, std::unique_ptr<Server>& server)
{
  InferenceServiceImpl service(
      &queue, &reference_outputs,
      std::vector<at::ScalarType>(
          expected_input_types.begin(), expected_input_types.end()));

  inference::GRPCInferenceService::AsyncService async_service;
  AsyncServerContext async_context(async_service, service);

  ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  async_context.configure(builder);
  const int grpc_max_message_bytes =
      max_message_bytes >
              static_cast<std::size_t>(std::numeric_limits<int>::max())
          ? std::numeric_limits<int>::max()
          : static_cast<int>(max_message_bytes);
  builder.SetMaxReceiveMessageSize(grpc_max_message_bytes);
  builder.SetMaxSendMessageSize(grpc_max_message_bytes);

  server = builder.BuildAndStart();
  if (!server) {
    log_error(std::format("Failed to start gRPC server on {}", address));
    return;
  }
  async_context.start();
  log_info(verbosity, std::format("Server listening on {}", address));
  server->Wait();
  async_context.shutdown();
  server.reset();
}

void
StopServer(std::unique_ptr<Server>& server)
{
  if (server) {
    server->Shutdown();
  }
}
}  // namespace starpu_server
