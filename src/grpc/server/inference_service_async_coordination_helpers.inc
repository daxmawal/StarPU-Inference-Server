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
    InferenceServiceImpl* service)
    -> std::optional<Status>
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
