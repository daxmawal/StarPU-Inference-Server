#include "inference_service_internal.hpp"

namespace starpu_server {

inline namespace inference_service_detail {

// Async submission/completion utility helpers.
auto
build_job_failure_result(InferenceJob& job)
    -> std::pair<Status, std::optional<InferenceServiceImpl::AsyncFailureInfo>>
{
  std::optional<InferenceServiceImpl::AsyncFailureInfo> failure_info;
  Status status = Status::OK;
  if (auto job_failure = job.take_failure_info()) {
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
    InferenceServiceImpl::AsyncFailureInfo info{};
    info.stage = std::move(job_failure->stage);
    info.reason = std::move(job_failure->reason);
    info.metrics_reported = job_failure->metrics_reported;
    failure_info = std::move(info);
  } else {
    InferenceServiceImpl::AsyncFailureInfo info{};
    info.stage = "execution";
    info.reason = "empty_output";
    failure_info = std::move(info);
    status = {grpc::StatusCode::INTERNAL, "Inference failed"};
  }
  return {status, std::move(failure_info)};
}

template <typename Callback>
void
handle_async_job_completion(
    InferenceJob& job, Callback&& callback, std::vector<torch::Tensor> outs,
    double latency_ms)
{
  const auto info = job.timing_info_snapshot();
  const auto base = detail::compute_latency_breakdown(info, latency_ms);
  InferenceServiceImpl::LatencyBreakdown timing{};
  timing.queue_ms = base.queue_ms;
  timing.batch_ms = base.batch_ms;
  timing.submit_ms = base.submit_ms;
  timing.scheduling_ms = base.scheduling_ms;
  timing.codelet_ms = base.codelet_ms;
  timing.inference_ms = base.inference_ms;
  timing.callback_ms = base.callback_ms;
  timing.total_ms = base.total_ms;
  detail::TimingInfo copied_info = info;

  if (outs.empty()) {
    auto [status, failure_info] = build_job_failure_result(job);
    std::forward<Callback>(callback)(
        status, {}, timing, copied_info, std::move(failure_info));
    return;
  }

  std::forward<Callback>(callback)(
      Status::OK, std::move(outs), timing, copied_info, std::nullopt);
}

void
set_submit_failure_info_if_needed(
    std::optional<InferenceServiceImpl::AsyncFailureInfo>* submit_failure_info,
    std::string_view reason)
{
  if (submit_failure_info == nullptr) {
    return;
  }
  InferenceServiceImpl::AsyncFailureInfo info{};
  info.stage = "enqueue";
  info.reason = std::string(reason);
  *submit_failure_info = std::move(info);
}

template <typename Callback>
void
report_async_completion_failure(
    const Callback& callback, std::string_view reason)
{
  InferenceServiceImpl::AsyncFailureInfo failure_info{
      .stage = "completion", .reason = std::string(reason)};
  try {
    callback(
        {grpc::StatusCode::INTERNAL,
         "Internal server error during async completion"},
        {}, InferenceServiceImpl::LatencyBreakdown{}, detail::TimingInfo{},
        std::move(failure_info));
  }
  catch (const std::exception& callback_exception) {
    log_error(std::format(
        "Unhandled exception while reporting async completion failure: {}",
        callback_exception.what()));
  }
  catch (...) {
    log_error(
        "Unhandled non-std exception while reporting async completion failure");
  }
}

void
dispatch_async_completion_safely(
    InferenceJob& job, const InferenceServiceImpl::AsyncJobCallback& callback,
    std::vector<torch::Tensor> outs, double latency_ms)
{
  try {
    handle_async_job_completion(job, callback, std::move(outs), latency_ms);
  }
  catch (const std::exception& e) {
    log_error(std::format(
        "Unhandled exception while dispatching async completion: {}",
        e.what()));
    report_async_completion_failure(callback, "exception");
  }
  catch (...) {
    log_error("Unhandled non-std exception while dispatching async completion");
    report_async_completion_failure(callback, "unknown_exception");
  }
}

auto
enqueue_inference_job(
    InferenceQueue& queue, const std::shared_ptr<InferenceJob>& job,
    std::optional<InferenceServiceImpl::AsyncFailureInfo>* submit_failure_info,
    InferenceServiceImpl* service) -> std::optional<Status>
{
  bool pushed = false;
  bool queue_full = false;
  {
    NvtxRange queue_scope("grpc_submit_starpu_queue");
    pushed = queue.push(job, &queue_full);
  }
  if (pushed) {
    return std::nullopt;
  }

  if (queue_full) {
    set_submit_failure_info_if_needed(submit_failure_info, "queue_full");
    if (service != nullptr) {
      if (auto metrics = service->metrics_recorder(); metrics != nullptr) {
        metrics->increment_rejected_requests();
      } else {
        increment_rejected_requests();
      }
      if (const auto& observability = service->observability();
          observability != nullptr &&
          observability->congestion_monitor != nullptr) {
        observability->congestion_monitor->record_rejection(1);
      } else {
        congestion::record_rejection(1);
      }
      if (const auto& observability = service->observability();
          observability != nullptr && observability->tracer != nullptr) {
        observability->tracer->log_request_rejected(queue.size());
      } else {
        BatchingTraceLogger::instance().log_request_rejected(queue.size());
      }
    } else {
      increment_rejected_requests();
      congestion::record_rejection(1);
      BatchingTraceLogger::instance().log_request_rejected(queue.size());
    }
    return Status{
        grpc::StatusCode::RESOURCE_EXHAUSTED, "Inference queue is full"};
  }

  set_submit_failure_info_if_needed(submit_failure_info, "queue_unavailable");
  return Status{grpc::StatusCode::UNAVAILABLE, "Inference queue unavailable"};
}

void
trace_enqueued_request_if_enabled(
    const std::shared_ptr<InferenceJob>& job, InferenceServiceImpl* service)
{
  auto* tracer = service != nullptr && service->observability() != nullptr &&
                         service->observability()->tracer != nullptr
                     ? service->observability()->tracer.get()
                     : &BatchingTraceLogger::instance();
  if (tracer->enabled()) {
    const auto timing = job->timing_info_snapshot();
    tracer->log_request_enqueued(
        job->get_request_id(), job->model_name(), /*is_warmup=*/false,
        timing.last_enqueued_time);
  }
}
}  // namespace inference_service_detail

// Async inference coordination block.
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

auto
InferenceServiceImpl::AsyncOps::build_latency_breakdown(
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
InferenceServiceImpl::
    submit_job_async(  // NOLINT(readability-function-cognitive-complexity)
        const std::vector<torch::Tensor>& inputs, AsyncJobCallback on_complete,
        std::vector<std::shared_ptr<const void>> input_lifetimes,
        std::shared_ptr<std::atomic<bool>> cancel_flag,
        MonotonicClock::time_point receive_time, std::string model_name,
        std::optional<AsyncFailureInfo>* submit_failure_info) -> Status
{
  auto resolved_model_name = resolve_model_name(std::move(model_name));
  if (submit_failure_info != nullptr) {
    *submit_failure_info = std::nullopt;
  }

  if (queue_ == nullptr) {
    set_submit_failure_info_if_needed(submit_failure_info, "queue_unavailable");
    return {grpc::StatusCode::UNAVAILABLE, "Inference queue unavailable"};
  }
  if (reference_outputs_ == nullptr) {
    set_submit_failure_info_if_needed(
        submit_failure_info, "reference_outputs_unavailable");
    return {
        grpc::StatusCode::FAILED_PRECONDITION,
        "Reference outputs are unavailable"};
  }

  try {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    auto& submit_hooks = testing::inference_service_test_internal::detail::
        submit_job_async_test_hooks_ref();
    if (submit_hooks.before_create_job) {
      submit_hooks.before_create_job();
    }
#endif  // SONAR_IGNORE_END
    auto job = client_utils::create_job(
        inputs, *reference_outputs_, next_request_id(),
        std::move(input_lifetimes), receive_time,
        std::string(resolved_model_name));
    job->set_cancelled_flag(std::move(cancel_flag));

    NvtxRange submit_scope("grpc_submit_starpu");

    job->set_on_complete(
        [job, callback = std::move(on_complete)](
            std::vector<torch::Tensor> outs, double latency_ms) mutable {
          dispatch_async_completion_safely(
              *job, callback, std::move(outs), latency_ms);
        });

    const auto enqueued_now = MonotonicClock::now();
    job->update_timing_info([enqueued_now](detail::TimingInfo& timing) {
      timing.enqueued_time = enqueued_now;
      timing.last_enqueued_time = enqueued_now;
    });

    if (auto enqueue_error =
            enqueue_inference_job(*queue_, job, submit_failure_info, this);
        enqueue_error.has_value()) {
      return *enqueue_error;
    }
    trace_enqueued_request_if_enabled(job, this);
    return Status::OK;
  }
  catch (const std::exception& e) {
    log_error(std::format(
        "Unhandled exception while submitting inference job: {}", e.what()));
    set_submit_failure_info_if_needed(submit_failure_info, "exception");
    return {grpc::StatusCode::INTERNAL, "Internal server error"};
  }
  catch (...) {
    log_error("Unhandled non-std exception while submitting inference job");
    set_submit_failure_info_if_needed(submit_failure_info, "unknown_exception");
    return {grpc::StatusCode::INTERNAL, "Internal server error"};
  }
}

void
InferenceServiceImpl::HandleModelInferAsync(
    ServerContext* context, const ModelInferRequest* request,
    ModelInferResponse* reply, std::function<void(Status)> on_done,
    std::shared_ptr<void> call_guard)
{
  auto callback_handle = std::make_shared<CallbackHandle>(std::move(on_done));
  auto cancel_flag = std::make_shared<std::atomic<bool>>(false);
  auto terminal_flag = std::make_shared<std::atomic<bool>>(false);
  const AsyncTerminalState terminal_state{
      .cancel_flag = cancel_flag, .terminal_flag = terminal_flag};
  AsyncOps::notify_cancel_flag_created(cancel_flag);
  if (auto guard_status = validate_model_infer_io(request, reply);
      !guard_status.ok()) {
    callback_handle->Invoke(guard_status);
    return;
  }

  const auto resolved_model_name = resolve_model_name(request->model_name());
  auto recv_tp = MonotonicClock::now();

  try {
    NvtxRange request_scope("grpc_handle_infer_request");

    auto metrics = metrics_recorder();
    if (metrics != nullptr) {
      metrics->increment_requests_total();
      metrics->increment_requests_received(resolved_model_name);
    } else {
      auto raw_metrics = get_metrics();
      if (raw_metrics && raw_metrics->counters().requests_total != nullptr) {
        raw_metrics->counters().requests_total->Increment();
      }
      increment_requests_received(resolved_model_name);
    }
    if (observability_ != nullptr &&
        observability_->congestion_monitor != nullptr) {
      observability_->congestion_monitor->record_arrival(1);
    } else {
      congestion::record_arrival(1);
    }
    int64_t recv_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
    if (AsyncOps::setup_async_cancellation(
            context, call_guard,
            AsyncCancellationContext{
                .cancel_flag = cancel_flag,
                .terminal_flag = terminal_flag,
                .callback_handle = callback_handle,
                .resolved_model_name = resolved_model_name,
                .service = this,
                .request = request,
                .recv_tp = recv_tp})) {
      return;
    }

    std::vector<torch::Tensor> inputs;
    std::vector<std::shared_ptr<const void>> input_lifetimes;
    std::optional<AsyncFailureInfo> submit_failure_info;
    Status status =
        validate_and_convert_inputs(request, inputs, &input_lifetimes);
    if (AsyncOps::handle_input_validation_failure(
            status, terminal_state, callback_handle, this, resolved_model_name,
            request, recv_tp)) {
      return;
    }

    if (cancel_flag->load(std::memory_order_acquire)) {
      return;
    }

    const auto* output_names = &expected_output_names_;
    const AsyncInferCompletionContext completion_context{
        .request = request,
        .reply = reply,
        .callback_handle = callback_handle,
        .metrics = metrics,
        .recv_tp = recv_tp,
        .recv_ms = recv_ms,
        .resolved_model_name = resolved_model_name,
        .impl = this,
        .output_names = output_names,
        .cancel_flag = cancel_flag,
        .terminal_flag = terminal_flag,
        .failure_info = std::nullopt};
    status = submit_job_async(
        inputs,
        [completion_context](
            Status const& job_status, const std::vector<torch::Tensor>& outs,
            LatencyBreakdown breakdown, detail::TimingInfo timing_info,
            std::optional<AsyncFailureInfo> failure_info) {
          AsyncOps::handle_submit_job_completion(
              completion_context, job_status, outs, breakdown, timing_info,
              std::move(failure_info));
        },
        std::move(input_lifetimes), cancel_flag, recv_tp, resolved_model_name,
        &submit_failure_info);

    AsyncOps::notify_submit_job_async_done(cancel_flag, status);
    if (AsyncOps::handle_submit_failure(
            status, terminal_state, callback_handle,
            AsyncTerminalCompletionDetails{
                .service = this,
                .resolved_model_name = resolved_model_name,
                .request = request,
                .recv_tp = recv_tp,
                .stage = "enqueue",
                .failure_info = &submit_failure_info,
            })) {
      return;
    }
  }
  catch (const std::exception& e) {
    AsyncOps::handle_async_internal_error(
        terminal_state, callback_handle, this, resolved_model_name, request,
        recv_tp,
        AsyncInternalErrorDetails{
            .stage = "internal",
            .reason = "exception",
            .log_context = std::format(
                "Unhandled exception in HandleModelInferAsync: {}", e.what())});
  }
  catch (...) {
    AsyncOps::handle_async_internal_error(
        terminal_state, callback_handle, this, resolved_model_name, request,
        recv_tp,
        AsyncInternalErrorDetails{
            .stage = "internal",
            .reason = "unknown_exception",
            .log_context = "Unhandled non-std exception in "
                           "HandleModelInferAsync"});
  }
}

auto
InferenceServiceImpl::AsyncOps::try_mark_terminal(
    const std::shared_ptr<std::atomic<bool>>& terminal_flag) -> bool
{
  if (terminal_flag == nullptr) {
    return true;
  }
  bool expected = false;
  return terminal_flag->compare_exchange_strong(
      expected, true, std::memory_order_acq_rel, std::memory_order_acquire);
}

auto
InferenceServiceImpl::AsyncOps::enter_async_terminal_once(
    const AsyncTerminalState& terminal_state, bool check_cancel_flag) -> bool
{
  if (check_cancel_flag && terminal_state.cancel_flag != nullptr &&
      terminal_state.cancel_flag->load(std::memory_order_acquire)) {
    return false;
  }
  return try_mark_terminal(terminal_state.terminal_flag);
}

auto
InferenceServiceImpl::AsyncOps::invoke_async_callback(
    const std::shared_ptr<CallbackHandle>& callback_handle,
    const Status& status) -> bool
{
  return callback_handle != nullptr && callback_handle->Invoke(status);
}

void
InferenceServiceImpl::AsyncOps::record_async_terminal_failure(
    const AsyncTerminalCompletionDetails& details, const Status& status)
{
  if (details.service != nullptr) {
    details.service->record_failure(
        details.request, details.recv_tp, details.resolved_model_name);
  }
  const auto no_failure_info = std::optional<AsyncFailureInfo>{};
  const auto& failure_info =
      details.failure_info != nullptr ? *details.failure_info : no_failure_info;
  if (details.service != nullptr) {
    details.service->record_terminal_metrics(
        details.resolved_model_name, status, details.stage, details.reason,
        failure_info, details.record_status_metric);
  }
}

auto
InferenceServiceImpl::AsyncOps::complete_async_terminal_with_status(
    const AsyncTerminalState& terminal_state,
    const std::shared_ptr<CallbackHandle>& callback_handle,
    const Status& status, const AsyncTerminalCompletionDetails& details) -> bool
{
  if (!enter_async_terminal_once(terminal_state, details.check_cancel_flag)) {
    return false;
  }

  const auto record_and_log = [&]() {
    record_async_terminal_failure(details, status);
    if (!details.log_context.empty()) {
      log_error(std::string(details.log_context));
    }
  };

  if (details.record_before_callback) {
    record_and_log();
    (void)invoke_async_callback(callback_handle, status);
    return true;
  }

  const bool callback_invoked = invoke_async_callback(callback_handle, status);
  if (details.require_callback_invoked_for_record && !callback_invoked) {
    return true;
  }
  record_and_log();
  return true;
}

auto
InferenceServiceImpl::AsyncOps::is_async_cancelled(
    const AsyncInferCompletionContext& context) -> bool
{
  return context.cancel_flag != nullptr &&
         context.cancel_flag->load(std::memory_order_acquire);
}

auto
InferenceServiceImpl::AsyncOps::prepare_async_completion(
    const AsyncInferCompletionContext& context,
    const std::shared_ptr<CallbackHandle>& callback_handle) -> bool
{
  if (is_async_cancelled(context)) {
    return false;
  }
  if (callback_handle == nullptr) {
    return false;
  }
  if (!callback_handle->TryAcquire()) {
    return false;
  }
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  auto& async_hooks = testing::inference_service_test_internal::detail::
      handle_async_infer_completion_test_hooks_ref();
  if (async_hooks.after_try_acquire && context.cancel_flag != nullptr) {
    async_hooks.after_try_acquire(context.cancel_flag);
  }
#endif  // SONAR_IGNORE_END
  if (is_async_cancelled(context)) {
    return false;
  }
  return true;
}

auto
InferenceServiceImpl::AsyncOps::handle_job_failure(
    const AsyncInferCompletionContext& context, const Status& job_status,
    const std::shared_ptr<CallbackHandle>& callback_handle) -> bool
{
  if (job_status.ok()) {
    return false;
  }

  (void)complete_async_terminal_with_status(
      AsyncTerminalState{context.cancel_flag, context.terminal_flag},
      callback_handle, job_status,
      AsyncTerminalCompletionDetails{
          .service = context.impl,
          .resolved_model_name = context.resolved_model_name,
          .request = context.request,
          .recv_tp = context.recv_tp,
          .stage = "execution",
          .failure_info = &context.failure_info,
          .check_cancel_flag = false,
      });
  return true;
}

void
InferenceServiceImpl::AsyncOps::finalize_successful_completion(
    const AsyncInferCompletionContext& context,
    const std::vector<torch::Tensor>& outs, LatencyBreakdown breakdown,
    const detail::TimingInfo& timing_info)
{
  const auto& callback_handle = context.callback_handle;
  if (callback_handle == nullptr) {
    const Status missing_callback_status{
        grpc::StatusCode::INTERNAL, "Internal server error"};
    (void)complete_async_terminal_with_status(
        AsyncTerminalState{context.cancel_flag, context.terminal_flag},
        callback_handle, missing_callback_status,
        AsyncTerminalCompletionDetails{
            .service = context.impl,
            .resolved_model_name = context.resolved_model_name,
            .request = context.request,
            .recv_tp = context.recv_tp,
            .stage = "postprocess",
            .reason = "missing_callback",
            .log_context =
                "Missing callback handle during async inference completion",
            .check_cancel_flag = false,
            .record_status_metric = false,
        });
    return;
  }
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  auto& async_hooks = testing::inference_service_test_internal::detail::
      handle_async_infer_completion_test_hooks_ref();
#endif  // SONAR_IGNORE_END

  const auto zero_tp = MonotonicClock::time_point{};
  if (timing_info.enqueued_time > zero_tp) {
    const auto preprocess_duration = std::chrono::duration<double, std::milli>(
        timing_info.enqueued_time - context.recv_tp);
    breakdown.preprocess_ms = std::max(0.0, preprocess_duration.count());
  } else {
    breakdown.preprocess_ms = 0.0;
  }

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (async_hooks.before_final_cancel_check && context.cancel_flag != nullptr) {
    async_hooks.before_final_cancel_check(context.cancel_flag);
  }
#endif  // SONAR_IGNORE_END
  if (is_async_cancelled(context)) {
    return;
  }

  const auto output_names =
      context.output_names != nullptr
          ? std::span<const std::string>(*context.output_names)
          : std::span<const std::string>{};
  InferenceServiceImpl::PopulateResponseOptions response_options;
  response_options.model_name_override = context.resolved_model_name;
  response_options.set_prepost_overall = false;
  response_options.output_names = output_names;
  Status populate_status = populate_response(
      context.request, context.reply, outs, context.recv_ms, breakdown,
      response_options);
  if (!populate_status.ok()) {
    (void)complete_async_terminal_with_status(
        AsyncTerminalState{context.cancel_flag, context.terminal_flag},
        callback_handle, populate_status,
        AsyncTerminalCompletionDetails{
            .service = context.impl,
            .resolved_model_name = context.resolved_model_name,
            .request = context.request,
            .recv_tp = context.recv_tp,
            .stage = "postprocess",
            .check_cancel_flag = false,
        });
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

  const auto latency_ms =
      std::chrono::duration<double, std::milli>(send_tp - context.recv_tp)
          .count();
  if (context.metrics && context.metrics->enabled()) {
    context.metrics->observe_inference_latency(latency_ms);
  } else {
    observe_inference_latency(latency_ms);
  }

  if (context.metrics != nullptr) {
    context.metrics->observe_latency_breakdown(LatencyBreakdownMetrics{
        breakdown.queue_ms,
        breakdown.batch_ms,
        breakdown.submit_ms,
        breakdown.scheduling_ms,
        breakdown.codelet_ms,
        breakdown.inference_ms,
        breakdown.callback_ms,
        breakdown.preprocess_ms,
        breakdown.postprocess_ms,
    });
  } else {
    observe_latency_breakdown(LatencyBreakdownMetrics{
        breakdown.queue_ms,
        breakdown.batch_ms,
        breakdown.submit_ms,
        breakdown.scheduling_ms,
        breakdown.codelet_ms,
        breakdown.inference_ms,
        breakdown.callback_ms,
        breakdown.preprocess_ms,
        breakdown.postprocess_ms,
    });
  }

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (async_hooks.before_success_terminal_mark &&
      context.cancel_flag != nullptr) {
    async_hooks.before_success_terminal_mark(context.cancel_flag);
  }
#endif  // SONAR_IGNORE_END
  if (is_async_cancelled(context)) {
    return;
  }
  if (!try_mark_terminal(context.terminal_flag)) {
    return;
  }

  if (context.impl != nullptr) {
    context.impl->record_success(
        context.request, breakdown, context.recv_tp,
        context.resolved_model_name);
  }
  if (context.impl != nullptr) {
    context.impl->record_terminal_metrics(
        context.resolved_model_name, Status::OK, "");
  }
  callback_handle->Invoke(Status::OK);
}

void
InferenceServiceImpl::AsyncOps::handle_async_infer_completion(
    const AsyncInferCompletionContext& context, const Status& job_status,
    const std::vector<torch::Tensor>& outs, LatencyBreakdown breakdown,
    detail::TimingInfo timing_info)
{
  const auto& callback_handle = context.callback_handle;
  if (!prepare_async_completion(context, callback_handle)) {
    return;
  }
  if (handle_job_failure(context, job_status, callback_handle)) {
    return;
  }
  finalize_successful_completion(context, outs, breakdown, timing_info);
}

void
InferenceServiceImpl::AsyncOps::handle_submit_job_completion(
    const AsyncInferCompletionContext& context, const Status& job_status,
    const std::vector<torch::Tensor>& outs, LatencyBreakdown breakdown,
    detail::TimingInfo timing_info,
    std::optional<AsyncFailureInfo> failure_info)
{
  try {
    auto completion_context = context;
    completion_context.failure_info = std::move(failure_info);
    handle_async_infer_completion(
        completion_context, job_status, outs, breakdown, timing_info);
  }
  catch (const std::exception& e) {
    handle_async_internal_error(
        AsyncTerminalState{context.cancel_flag, context.terminal_flag},
        context.callback_handle, context.impl, context.resolved_model_name,
        context.request, context.recv_tp,
        AsyncInternalErrorDetails{
            .stage = "postprocess",
            .reason = "exception",
            .log_context = std::format(
                "Unhandled exception in async inference completion: {}",
                e.what())});
  }
  catch (...) {
    handle_async_internal_error(
        AsyncTerminalState{context.cancel_flag, context.terminal_flag},
        context.callback_handle, context.impl, context.resolved_model_name,
        context.request, context.recv_tp,
        AsyncInternalErrorDetails{
            .stage = "postprocess",
            .reason = "unknown_exception",
            .log_context =
                "Unhandled non-std exception in async inference completion"});
  }
}

void
InferenceServiceImpl::AsyncOps::notify_cancel_flag_created(
    const std::shared_ptr<std::atomic<bool>>& cancel_flag)
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  auto& test_hooks = testing::inference_service_test_internal::detail::
      handle_model_infer_async_test_hooks_ref();
  if (test_hooks.on_cancel_flag_created) {
    test_hooks.on_cancel_flag_created(cancel_flag);
  }
#else
  (void)cancel_flag;
#endif  // SONAR_IGNORE_END
}

void
InferenceServiceImpl::AsyncOps::notify_submit_job_async_done(
    const std::shared_ptr<std::atomic<bool>>& cancel_flag, const Status& status)
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  auto& test_hooks = testing::inference_service_test_internal::detail::
      handle_model_infer_async_test_hooks_ref();
  if (test_hooks.on_submit_job_async_done) {
    test_hooks.on_submit_job_async_done(cancel_flag, status);
  }
#else
  (void)cancel_flag;
  (void)status;
#endif  // SONAR_IGNORE_END
}

auto
InferenceServiceImpl::AsyncOps::setup_async_cancellation(
    ServerContext* context, std::shared_ptr<void>& call_guard,
    const AsyncCancellationContext& cancellation_context) -> bool
{
  if (context == nullptr || !call_guard ||
      cancellation_context.cancel_flag == nullptr ||
      cancellation_context.callback_handle == nullptr) {
    return false;
  }

  const AsyncTerminalState terminal_state{
      cancellation_context.cancel_flag, cancellation_context.terminal_flag};

  auto on_cancel = [context, terminal_state,
                    callback_handle = cancellation_context.callback_handle,
                    model_name =
                        std::string(cancellation_context.resolved_model_name),
                    service = cancellation_context.service,
                    request = cancellation_context.request,
                    recv_tp = cancellation_context.recv_tp]() {
    handle_async_cancellation(
        context, terminal_state, callback_handle, service, model_name, request,
        recv_tp);
  };

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  auto& test_hooks = testing::inference_service_test_internal::detail::
      handle_model_infer_async_test_hooks_ref();
  if (test_hooks.on_cancel_ready) {
    test_hooks.on_cancel_ready(on_cancel);
  }
#endif  // SONAR_IGNORE_END

  auto done_tag = RpcDoneTag::Create(on_cancel, std::move(call_guard));
  done_tag->Arm(context);
  if (is_context_cancelled(context)) {
    on_cancel();
    return true;
  }
  return false;
}

void
InferenceServiceImpl::AsyncOps::handle_async_cancellation(
    const ServerContext* context, const AsyncTerminalState& terminal_state,
    const std::shared_ptr<CallbackHandle>& callback_handle,
    InferenceServiceImpl* service, std::string_view resolved_model_name,
    const ModelInferRequest* request, MonotonicClock::time_point recv_tp)
{
  if (!is_context_cancelled(context) || terminal_state.cancel_flag == nullptr) {
    return;
  }
  if (terminal_state.cancel_flag->exchange(true, std::memory_order_acq_rel)) {
    return;
  }

  const Status cancelled_status{
      grpc::StatusCode::CANCELLED, "Request cancelled"};
  (void)complete_async_terminal_with_status(
      terminal_state, callback_handle, cancelled_status,
      AsyncTerminalCompletionDetails{
          .service = service,
          .resolved_model_name = resolved_model_name,
          .request = request,
          .recv_tp = recv_tp,
          .stage = "cancel",
          .reason = "client_cancelled",
          .check_cancel_flag = false,
          .record_before_callback = false,
          .require_callback_invoked_for_record = true,
      });
}

auto
InferenceServiceImpl::AsyncOps::handle_input_validation_failure(
    const Status& status,
    const InferenceServiceImpl::AsyncTerminalState& terminal_state,
    const std::shared_ptr<CallbackHandle>& callback_handle,
    InferenceServiceImpl* service, std::string_view resolved_model_name,
    const ModelInferRequest* request,
    MonotonicClock::time_point recv_tp) -> bool
{
  if (status.ok()) {
    return false;
  }
  (void)complete_async_terminal_with_status(
      terminal_state, callback_handle, status,
      AsyncTerminalCompletionDetails{
          .service = service,
          .resolved_model_name = resolved_model_name,
          .request = request,
          .recv_tp = recv_tp,
          .stage = "preprocess",
      });
  return true;
}

auto
InferenceServiceImpl::AsyncOps::handle_submit_failure(
    const Status& status,
    const InferenceServiceImpl::AsyncTerminalState& terminal_state,
    const std::shared_ptr<CallbackHandle>& callback_handle,
    const InferenceServiceImpl::AsyncTerminalCompletionDetails& details) -> bool
{
  if (status.ok()) {
    return false;
  }
  (void)complete_async_terminal_with_status(
      terminal_state, callback_handle, status, details);
  return true;
}

void
InferenceServiceImpl::AsyncOps::handle_async_internal_error(
    const InferenceServiceImpl::AsyncTerminalState& terminal_state,
    const std::shared_ptr<CallbackHandle>& callback_handle,
    InferenceServiceImpl* service, std::string_view resolved_model_name,
    const ModelInferRequest* request, MonotonicClock::time_point recv_tp,
    const InferenceServiceImpl::AsyncInternalErrorDetails& details)
{
  const Status status{grpc::StatusCode::INTERNAL, "Internal server error"};
  (void)complete_async_terminal_with_status(
      terminal_state, callback_handle, status,
      AsyncTerminalCompletionDetails{
          .service = service,
          .resolved_model_name = resolved_model_name,
          .request = request,
          .recv_tp = recv_tp,
          .stage = details.stage,
          .reason = details.reason,
          .log_context = details.log_context,
          .record_before_callback = false,
          .require_callback_invoked_for_record = true,
      });
}

void
InferenceServiceImpl::record_success(
    const ModelInferRequest* request, const LatencyBreakdown& breakdown,
    MonotonicClock::time_point recv_tp, std::string_view resolved_model_name)
{
  if (request == nullptr) {
    return;
  }
  const auto now_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  const uint64_t batch_size = request_batch_size(request, max_batch_size_);
  const uint64_t total_ns = duration_ms_to_ns(
      breakdown.overall_ms > 0.0 ? breakdown.overall_ms : breakdown.total_ms);
  (void)recv_tp;

  ModelStatsKey key{std::string(resolved_model_name), request->model_version()};
  std::scoped_lock lock(model_stats_mutex_);
  auto& stats = model_stats_[key];
  stats.last_inference_ms = now_ms;
  stats.inference_count += batch_size;
  stats.execution_count += 1;
  stats.inference_stats.success.count += 1;
  stats.inference_stats.success.ns += total_ns;
  stats.inference_stats.queue.count += 1;
  stats.inference_stats.queue.ns += duration_ms_to_ns(breakdown.queue_ms);
  stats.inference_stats.compute_input.count += 1;
  stats.inference_stats.compute_input.ns +=
      duration_ms_to_ns(breakdown.preprocess_ms);
  stats.inference_stats.compute_infer.count += 1;
  stats.inference_stats.compute_infer.ns +=
      duration_ms_to_ns(breakdown.inference_ms);
  stats.inference_stats.compute_output.count += 1;
  stats.inference_stats.compute_output.ns +=
      duration_ms_to_ns(breakdown.postprocess_ms);
}

void
InferenceServiceImpl::record_failure(
    const ModelInferRequest* request, MonotonicClock::time_point recv_tp,
    std::string_view resolved_model_name)
{
  if (request == nullptr) {
    return;
  }
  const auto now_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  const uint64_t total_ns = elapsed_since(recv_tp);

  ModelStatsKey key{std::string(resolved_model_name), request->model_version()};
  std::scoped_lock lock(model_stats_mutex_);
  auto& stats = model_stats_[key];
  stats.last_inference_ms = now_ms;
  stats.inference_stats.fail.count += 1;
  stats.inference_stats.fail.ns += total_ns;
}

}  // namespace starpu_server
