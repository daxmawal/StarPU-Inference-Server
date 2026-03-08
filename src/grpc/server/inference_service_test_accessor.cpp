#if !defined(STARPU_TESTING)
#error \
    "inference_service_test_accessor.cpp is test-only and requires STARPU_TESTING"
#endif

#include "core/inference_runner.hpp"
#include "inference_service_test_internal.hpp"

namespace starpu_server::testing {

void
InferenceServiceTestAccessor::SetHandleModelInferAsyncTestHooks(
    HandleModelInferAsyncTestHooks hooks)
{
  inference_service_test_internal::set_handle_model_infer_async_test_hooks(
      std::move(hooks));
}

void
InferenceServiceTestAccessor::ClearHandleModelInferAsyncTestHooks()
{
  inference_service_test_internal::clear_handle_model_infer_async_test_hooks();
}

void
InferenceServiceTestAccessor::SetHandleAsyncInferCompletionTestHooks(
    HandleAsyncInferCompletionTestHooks hooks)
{
  inference_service_test_internal::set_handle_async_infer_completion_test_hooks(
      std::move(hooks));
}

void
InferenceServiceTestAccessor::ClearHandleAsyncInferCompletionTestHooks()
{
  inference_service_test_internal::
      clear_handle_async_infer_completion_test_hooks();
}

void
InferenceServiceTestAccessor::SetSubmitJobAsyncTestHooks(
    SubmitJobAsyncTestHooks hooks)
{
  inference_service_test_internal::set_submit_job_async_test_hooks(
      std::move(hooks));
}

void
InferenceServiceTestAccessor::ClearSubmitJobAsyncTestHooks()
{
  inference_service_test_internal::clear_submit_job_async_test_hooks();
}

auto
InferenceServiceTestAccessor::NormalizeNamesForTest(
    std::vector<std::string> names, std::size_t expected_size,
    std::string_view fallback_prefix,
    std::string_view kind) -> std::vector<std::string>
{
  return inference_service_test_internal::normalize_names_for_test(
      std::move(names), expected_size, fallback_prefix, kind);
}

void
InferenceServiceTestAccessor::SetExpectedInputTypesForTest(
    InferenceServiceImpl* service, std::vector<at::ScalarType> types)
{
  service->expected_input_types_ = std::move(types);
}

void
InferenceServiceTestAccessor::SetExpectedInputNamesForTest(
    InferenceServiceImpl* service, std::vector<std::string> names)
{
  service->expected_input_names_ = std::move(names);
}

void
InferenceServiceTestAccessor::SetReferenceOutputsForTest(
    InferenceServiceImpl* service,
    const std::vector<torch::Tensor>* reference_outputs)
{
  service->reference_outputs_ = reference_outputs;
}

auto
InferenceServiceTestAccessor::CheckMissingInputsForTest(
    const std::vector<bool>& filled,
    std::span<const std::string> expected_names) -> grpc::Status
{
  return inference_service_test_internal::check_missing_named_inputs_for_test(
      filled, expected_names);
}

void
InferenceServiceTestAccessor::ArmRpcDoneTagWithNullContextForTest()
{
  inference_service_test_internal::
      arm_rpc_done_tag_with_null_context_for_test();
}

auto
InferenceServiceTestAccessor::RpcDoneTagProceedForTest(
    bool is_ok, bool with_on_done) -> bool
{
  return inference_service_test_internal::rpc_done_tag_proceed_for_test(
      is_ok, with_on_done);
}

void
InferenceServiceTestAccessor::SetGrpcHealthStatusForTest(
    grpc::Server* server, bool serving)
{
  inference_service_test_internal::set_grpc_health_status_for_test(
      server, serving);
}

void
InferenceServiceTestAccessor::RecordSuccessForTest(
    InferenceServiceImpl* service, const inference::ModelInferRequest* request,
    const InferenceServiceImpl::LatencyBreakdown& breakdown,
    MonotonicClock::time_point recv_tp, std::string_view resolved_model_name)
{
  service->record_success(request, breakdown, recv_tp, resolved_model_name);
}

void
InferenceServiceTestAccessor::RecordFailureForTest(
    InferenceServiceImpl* service, const inference::ModelInferRequest* request,
    MonotonicClock::time_point recv_tp, std::string_view resolved_model_name)
{
  service->record_failure(request, recv_tp, resolved_model_name);
}

auto
InferenceServiceTestAccessor::ScalarTypeToModelDtypeForTest(at::ScalarType type)
    -> inference::DataType
{
  return inference_service_test_internal::scalar_type_to_model_dtype_for_test(
      type);
}

auto
InferenceServiceTestAccessor::ResolveTensorNameForTest(
    std::size_t index, std::span<const std::string> names,
    std::string_view fallback_prefix) -> std::string
{
  return inference_service_test_internal::resolve_tensor_name_for_test(
      index, names, fallback_prefix);
}

auto
InferenceServiceTestAccessor::RequestBatchSizeForTest(
    const inference::ModelInferRequest* request, int max_batch_size) -> uint64_t
{
  return inference_service_test_internal::request_batch_size_for_test(
      request, max_batch_size);
}

auto
InferenceServiceTestAccessor::DurationMsToNsForTest(double duration_ms)
    -> uint64_t
{
  return inference_service_test_internal::duration_ms_to_ns_for_test(
      duration_ms);
}

auto
InferenceServiceTestAccessor::ElapsedSinceForTest(
    MonotonicClock::time_point start) -> uint64_t
{
  return inference_service_test_internal::elapsed_since_for_test(start);
}

auto
InferenceServiceTestAccessor::ResolveTerminalFailureStageForTest(
    const grpc::Status& status, std::string_view default_stage,
    std::string_view default_reason,
    const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info)
    -> std::string
{
  return inference_service_test_internal::
      resolve_terminal_failure_stage_for_test(
          status, default_stage, default_reason, failure_info);
}

auto
InferenceServiceTestAccessor::ShouldReportTerminalFailureMetricForTest(
    const grpc::Status& status, std::string_view default_stage,
    std::string_view default_reason,
    const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info)
    -> bool
{
  return inference_service_test_internal::
      should_report_terminal_failure_metric_for_test(
          status, default_stage, default_reason, failure_info);
}

void
InferenceServiceTestAccessor::SetModelStatisticsForceNullTargetForTest(
    bool enable)
{
  inference_service_test_internal::
      set_model_statistics_force_null_target_for_test(enable);
}

auto
InferenceServiceTestAccessor::IsContextCancelledForTest(
    grpc::ServerContext* context) -> bool
{
  return inference_service_test_internal::is_context_cancelled_for_test(
      context);
}

auto
InferenceServiceTestAccessor::FillOutputTensorForTest(
    inference::ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs,
    const std::vector<std::size_t>& output_indices,
    const std::vector<std::string>& output_names) -> grpc::Status
{
  return inference_service_test_internal::fill_output_tensor_for_test(
      reply, outputs, output_indices, output_names);
}

auto
InferenceServiceTestAccessor::BuildLatencyBreakdownForTest(
    const starpu_server::detail::TimingInfo& info,
    double latency_ms) -> InferenceServiceImpl::LatencyBreakdown
{
  return InferenceServiceImpl::build_latency_breakdown(info, latency_ms);
}

auto
InferenceServiceTestAccessor::HandleSubmitFailureForTest(
    const grpc::Status& status, bool cancelled, bool terminal_marked,
    std::atomic<bool>* callback_invoked,
    const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info)
    -> bool
{
  auto cancel_flag = std::make_shared<std::atomic<bool>>(cancelled);
  auto terminal_flag = std::make_shared<std::atomic<bool>>(terminal_marked);
  std::shared_ptr<InferenceServiceImpl::CallbackHandle> callback_handle;
  if (callback_invoked != nullptr) {
    callback_handle = std::make_shared<InferenceServiceImpl::CallbackHandle>(
        [callback_invoked](grpc::Status /*unused*/) {
          callback_invoked->store(true, std::memory_order_release);
        });
  }
  return InferenceServiceImpl::handle_submit_failure(
      status,
      InferenceServiceImpl::AsyncTerminalState{cancel_flag, terminal_flag},
      callback_handle, nullptr, "model", nullptr, MonotonicClock::now(),
      failure_info);
}

auto
InferenceServiceTestAccessor::HandleInputValidationFailureForTest(
    const grpc::Status& status, bool cancelled, bool terminal_marked,
    std::atomic<bool>* callback_invoked) -> bool
{
  auto cancel_flag = std::make_shared<std::atomic<bool>>(cancelled);
  auto terminal_flag = std::make_shared<std::atomic<bool>>(terminal_marked);
  std::shared_ptr<InferenceServiceImpl::CallbackHandle> callback_handle;
  if (callback_invoked != nullptr) {
    callback_handle = std::make_shared<InferenceServiceImpl::CallbackHandle>(
        [callback_invoked](grpc::Status /*unused*/) {
          callback_invoked->store(true, std::memory_order_release);
        });
  }
  return InferenceServiceImpl::handle_input_validation_failure(
      status,
      InferenceServiceImpl::AsyncTerminalState{cancel_flag, terminal_flag},
      callback_handle, nullptr, "model", nullptr, MonotonicClock::now());
}

auto
InferenceServiceTestAccessor::FinalizeSuccessfulCompletionForTest(
    bool cancelled, bool terminal_marked, bool callback_present,
    bool reply_present, bool with_impl,
    std::atomic<bool>* callback_invoked) -> bool
{
  if (callback_invoked != nullptr) {
    callback_invoked->store(false, std::memory_order_release);
  }

  inference::ModelInferRequest request;
  inference::ModelInferResponse reply;
  std::vector<torch::Tensor> outputs = {
      torch::zeros({1}, torch::TensorOptions().dtype(at::kFloat))};
  auto cancel_flag = std::make_shared<std::atomic<bool>>(cancelled);
  auto terminal_flag = std::make_shared<std::atomic<bool>>(terminal_marked);

  std::shared_ptr<InferenceServiceImpl::CallbackHandle> callback_handle;
  if (callback_present) {
    callback_handle = std::make_shared<InferenceServiceImpl::CallbackHandle>(
        [callback_invoked](grpc::Status /*unused*/) {
          if (callback_invoked != nullptr) {
            callback_invoked->store(true, std::memory_order_release);
          }
        });
  }

  InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  std::unique_ptr<InferenceServiceImpl> service;
  if (with_impl) {
    service = std::make_unique<InferenceServiceImpl>(
        &queue, &reference_outputs, std::vector<at::ScalarType>{at::kFloat});
  }

  InferenceServiceImpl::LatencyBreakdown breakdown{};
  starpu_server::detail::TimingInfo timing_info{};
  const InferenceServiceImpl::AsyncInferCompletionContext context{
      .request = &request,
      .reply = reply_present ? &reply : nullptr,
      .callback_handle = callback_handle,
      .metrics = nullptr,
      .recv_tp = MonotonicClock::now(),
      .recv_ms = 0,
      .resolved_model_name = "model",
      .impl = service.get(),
      .output_names = nullptr,
      .cancel_flag = cancel_flag,
      .terminal_flag = terminal_flag,
      .failure_info = std::nullopt};
  InferenceServiceImpl::finalize_successful_completion(
      context, outputs, breakdown, timing_info);
  return terminal_flag->load(std::memory_order_acquire);
}

auto
InferenceServiceTestAccessor::HandleJobFailureForTest(
    const grpc::Status& job_status, bool terminal_marked, bool callback_present,
    std::atomic<bool>* callback_invoked) -> bool
{
  if (callback_invoked != nullptr) {
    callback_invoked->store(false, std::memory_order_release);
  }

  inference::ModelInferRequest request;
  inference::ModelInferResponse reply;
  auto cancel_flag = std::make_shared<std::atomic<bool>>(false);
  auto terminal_flag = std::make_shared<std::atomic<bool>>(terminal_marked);

  std::shared_ptr<InferenceServiceImpl::CallbackHandle> callback_handle;
  if (callback_present) {
    callback_handle = std::make_shared<InferenceServiceImpl::CallbackHandle>(
        [callback_invoked](grpc::Status /*unused*/) {
          if (callback_invoked != nullptr) {
            callback_invoked->store(true, std::memory_order_release);
          }
        });
  }

  const InferenceServiceImpl::AsyncInferCompletionContext context{
      .request = &request,
      .reply = &reply,
      .callback_handle = callback_handle,
      .metrics = nullptr,
      .recv_tp = MonotonicClock::now(),
      .recv_ms = 0,
      .resolved_model_name = "model",
      .impl = nullptr,
      .output_names = nullptr,
      .cancel_flag = cancel_flag,
      .terminal_flag = terminal_flag,
      .failure_info = std::nullopt};
  return InferenceServiceImpl::handle_job_failure(
      context, job_status, callback_handle);
}

auto
InferenceServiceTestAccessor::PrepareAsyncCompletionForTest(
    bool cancelled, bool callback_present) -> bool
{
  inference::ModelInferRequest request;
  inference::ModelInferResponse reply;
  auto cancel_flag = std::make_shared<std::atomic<bool>>(cancelled);
  auto terminal_flag = std::make_shared<std::atomic<bool>>(false);

  std::shared_ptr<InferenceServiceImpl::CallbackHandle> callback_handle;
  if (callback_present) {
    callback_handle = std::make_shared<InferenceServiceImpl::CallbackHandle>(
        [](grpc::Status /*unused*/) {});
  }

  const InferenceServiceImpl::AsyncInferCompletionContext context{
      .request = &request,
      .reply = &reply,
      .callback_handle = callback_handle,
      .metrics = nullptr,
      .recv_tp = MonotonicClock::now(),
      .recv_ms = 0,
      .resolved_model_name = "model",
      .impl = nullptr,
      .output_names = nullptr,
      .cancel_flag = cancel_flag,
      .terminal_flag = terminal_flag,
      .failure_info = std::nullopt};
  return InferenceServiceImpl::prepare_async_completion(
      context, callback_handle);
}

auto
InferenceServiceTestAccessor::TryMarkTerminalNullFlagForTest() -> bool
{
  const std::shared_ptr<std::atomic<bool>> terminal_flag;
  return InferenceServiceImpl::try_mark_terminal(terminal_flag);
}

auto
InferenceServiceTestAccessor::HandleAsyncInferCompletionForTest(bool cancelled)
    -> bool
{
  inference::ModelInferRequest request;
  inference::ModelInferResponse reply;
  std::vector<torch::Tensor> outputs = {
      torch::zeros({1}, torch::TensorOptions().dtype(at::kFloat))};
  auto cancel_flag = std::make_shared<std::atomic<bool>>(cancelled);
  auto terminal_flag = std::make_shared<std::atomic<bool>>(false);
  bool called = false;
  auto callback_handle = std::make_shared<InferenceServiceImpl::CallbackHandle>(
      [&called](grpc::Status /*unused*/) { called = true; });
  InferenceServiceImpl::LatencyBreakdown breakdown{};
  starpu_server::detail::TimingInfo timing_info{};
  InferenceServiceImpl::AsyncInferCompletionContext context{
      &request,
      &reply,
      callback_handle,
      nullptr,
      MonotonicClock::now(),
      0,
      "model",
      nullptr,
      nullptr,
      cancel_flag,
      terminal_flag,
      std::nullopt};
  InferenceServiceImpl::handle_async_infer_completion(
      context, grpc::Status::OK, outputs, breakdown, timing_info);
  return called;
}

auto
InferenceServiceTestAccessor::ValidateConfiguredShapeForTest(
    const std::vector<int64_t>& shape, const std::vector<int64_t>& expected,
    bool batching_allowed, int max_batch_size) -> grpc::Status
{
  return inference_service_test_internal::validate_configured_shape_for_test(
      shape, expected, batching_allowed, max_batch_size);
}

auto
InferenceServiceTestAccessor::
    UnaryCallDataMissingHandlerTransitionsToFinishForTest() -> bool
{
  return inference_service_test_internal::
      unary_call_data_missing_handler_transitions_to_finish_for_test();
}

}  // namespace starpu_server::testing
