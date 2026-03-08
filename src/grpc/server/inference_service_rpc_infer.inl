auto
InferenceServiceImpl::validate_and_convert_inputs(
    const ModelInferRequest* request, std::vector<torch::Tensor>& inputs,
    std::vector<std::shared_ptr<const void>>* input_lifetimes) -> Status
{
  if (auto status =
          validate_input_counts(request, expected_input_types_.size());
      !status.ok()) {
    return status;
  }

  const auto name_state = collect_input_name_state(request);
  const bool has_expected_names = !expected_input_names_.empty();
  if (auto status = validate_input_names(name_state, has_expected_names);
      !status.ok()) {
    return status;
  }
  const bool use_name_mapping = has_expected_names;

  std::unordered_map<std::string_view, std::size_t> expected_index_by_name;
  if (use_name_mapping) {
    if (auto status = build_expected_index_by_name(
            expected_input_names_, expected_index_by_name);
        !status.ok()) {
      return status;
    }
  }

  const std::size_t expected_count = expected_input_types_.size();
  std::vector<torch::Tensor> ordered_inputs(expected_count);
  std::vector<std::shared_ptr<const void>> ordered_lifetimes;
  if (input_lifetimes != nullptr) {
    ordered_lifetimes.resize(expected_count);
  }
  std::vector<bool> filled(expected_count, false);
  const ProcessInputContext process_context{
      use_name_mapping ? &expected_index_by_name : nullptr,
      &expected_input_types_,
      &expected_input_dims_,
      max_batch_size_,
      &ordered_inputs,
      input_lifetimes != nullptr ? &ordered_lifetimes : nullptr,
      &filled};

  for (int i = 0; i < request->inputs_size(); ++i) {
    Status status = process_input(request, i, process_context);
    if (!status.ok()) {
      return status;
    }
  }

  if (use_name_mapping) {
    Status status = check_missing_named_inputs(filled, expected_input_names_);
    if (!status.ok()) {
      return status;
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
            enqueue_inference_job(*queue_, job, submit_failure_info);
        enqueue_error.has_value()) {
      return *enqueue_error;
    }
    trace_enqueued_request_if_enabled(job);
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
  notify_cancel_flag_created(cancel_flag);
  if (auto guard_status = validate_model_infer_io(request, reply);
      !guard_status.ok()) {
    callback_handle->Invoke(guard_status);
    return;
  }

  const auto resolved_model_name = resolve_model_name(request->model_name());
  auto recv_tp = MonotonicClock::now();

  try {
    NvtxRange request_scope("grpc_handle_infer_request");

    auto metrics = get_metrics();
    if (metrics && metrics->counters().requests_total != nullptr) {
      metrics->counters().requests_total->Increment();
    }
    increment_requests_received(resolved_model_name);
    congestion::record_arrival(1);
    int64_t recv_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
    if (setup_async_cancellation(
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
    if (handle_input_validation_failure(
            status, terminal_state, callback_handle, this, resolved_model_name,
            request, recv_tp)) {
      return;
    }

    if (cancel_flag->load(std::memory_order_acquire)) {
      return;
    }

    const auto* output_names = &expected_output_names_;
    status = submit_job_async(
        inputs,
        [request, reply, recv_tp, recv_ms, metrics, callback_handle,
         resolved_model_name, cancel_flag, terminal_flag, terminal_state,
         output_names, service = this](
            Status const& job_status, const std::vector<torch::Tensor>& outs,
            LatencyBreakdown breakdown, detail::TimingInfo timing_info,
            std::optional<AsyncFailureInfo> failure_info) mutable {
          try {
            handle_async_infer_completion(
                AsyncInferCompletionContext{
                    request, reply, callback_handle, metrics, recv_tp, recv_ms,
                    resolved_model_name, service, output_names, cancel_flag,
                    terminal_flag, std::move(failure_info)},
                job_status, outs, breakdown, timing_info);
          }
          catch (const std::exception& e) {
            handle_async_internal_error(
                terminal_state, callback_handle, service, resolved_model_name,
                request, recv_tp,
                AsyncInternalErrorDetails{
                    .stage = "postprocess",
                    .reason = "exception",
                    .log_context = std::format(
                        "Unhandled exception in async inference completion: {}",
                        e.what())});
          }
          catch (...) {
            handle_async_internal_error(
                terminal_state, callback_handle, service, resolved_model_name,
                request, recv_tp,
                AsyncInternalErrorDetails{
                    .stage = "postprocess",
                    .reason = "unknown_exception",
                    .log_context =
                        "Unhandled non-std exception in async inference "
                        "completion"});
          }
        },
        std::move(input_lifetimes), cancel_flag, recv_tp, resolved_model_name,
        &submit_failure_info);

    notify_submit_job_async_done(cancel_flag, status);
    if (handle_submit_failure(
            status, terminal_state, callback_handle, this, resolved_model_name,
            request, recv_tp, submit_failure_info)) {
      return;
    }
  }
  catch (const std::exception& e) {
    handle_async_internal_error(
        terminal_state, callback_handle, this, resolved_model_name, request,
        recv_tp,
        AsyncInternalErrorDetails{
            .stage = "internal",
            .reason = "exception",
            .log_context = std::format(
                "Unhandled exception in HandleModelInferAsync: {}", e.what())});
  }
  catch (...) {
    handle_async_internal_error(
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
InferenceServiceImpl::populate_response(
    const ModelInferRequest* request, ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs, int64_t recv_ms,
    const LatencyBreakdown& breakdown) -> Status
{
  return populate_response(
      request, reply, outputs, recv_ms, breakdown, PopulateResponseOptions{});
}

auto
InferenceServiceImpl::populate_response(
    const ModelInferRequest* request, ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs, int64_t recv_ms,
    const LatencyBreakdown& breakdown,
    PopulateResponseOptions options) -> Status
{
  if (request == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "ModelInfer request is null"};
  }
  if (reply == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "ModelInfer response is null"};
  }
  if (!options.model_name_override.empty()) {
    reply->set_model_name(std::string(options.model_name_override));
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
  if (options.set_prepost_overall) {
    reply->set_server_preprocess_ms(breakdown.preprocess_ms);
    reply->set_server_postprocess_ms(breakdown.postprocess_ms);
    reply->set_server_overall_ms(breakdown.overall_ms);
  }
  reply->set_server_total_ms(breakdown.total_ms);

  const auto resolved_names =
      resolve_output_names(options.output_names, outputs.size());
  std::vector<std::size_t> output_indices;
  if (request->outputs_size() > 0) {
    std::unordered_map<std::string_view, std::size_t> index_by_name;
    index_by_name.reserve(resolved_names.size());
    for (std::size_t i = 0; i < resolved_names.size(); ++i) {
      const auto [existing_it, inserted] =
          index_by_name.try_emplace(resolved_names[i], i);
      if (!inserted) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format(
                "Configured output name '{}' is duplicated",
                existing_it->first)};
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
      const auto name_iter = index_by_name.find(requested.name());
      if (name_iter == index_by_name.end()) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format(
                "Requested output '{}' is not available", requested.name())};
      }
      if (!seen.insert(name_iter->first).second) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format(
                "Requested output '{}' is duplicated", requested.name())};
      }
      output_indices.push_back(name_iter->second);
    }
  } else {
    output_indices.reserve(outputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      output_indices.push_back(i);
    }
  }

  return fill_output_tensor(reply, outputs, output_indices, resolved_names);
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
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
#endif  // SONAR_IGNORE_END
