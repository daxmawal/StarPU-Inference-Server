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

void
InferenceServiceTestAccessor::SetCheckMissingNamedInputsOverrideForTest(
    CheckMissingNamedInputsOverrideFn override_fn)
{
  inference_service_test_internal::set_check_missing_named_inputs_override(
      std::move(override_fn));
}

void
InferenceServiceTestAccessor::ClearCheckMissingNamedInputsOverrideForTest()
{
  inference_service_test_internal::clear_check_missing_named_inputs_override();
}

#define STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET(                        \
    return_type, accessor_name, internal_name, runtime_name, params_decl,   \
    args)                                                                   \
  auto InferenceServiceTestAccessor::accessor_name params_decl->return_type \
  {                                                                         \
    return inference_service_test_internal::internal_name args;             \
  }

#define STARPU_INFERENCE_SERVICE_TEST_FORWARDER_VOID(              \
    accessor_name, internal_name, runtime_name, params_decl, args) \
  void InferenceServiceTestAccessor::accessor_name params_decl     \
  {                                                                \
    inference_service_test_internal::internal_name args;           \
  }

#include "inference_service_test_forwarders.hpp"

#undef STARPU_INFERENCE_SERVICE_TEST_FORWARDER_VOID
#undef STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET

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

void
InferenceServiceTestAccessor::SetModelStatisticsForceNullTargetForTest(
    bool enable)
{
  inference_service_test_internal::
      set_model_statistics_force_null_target_for_test(enable);
}

auto
InferenceServiceTestAccessor::BuildLatencyBreakdownForTest(
    const starpu_server::detail::TimingInfo& info,
    double latency_ms) -> InferenceServiceImpl::LatencyBreakdown
{
  return InferenceServiceImpl::AsyncOps::build_latency_breakdown(
      info, latency_ms);
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
  return InferenceServiceImpl::AsyncOps::handle_submit_failure(
      status,
      InferenceServiceImpl::AsyncTerminalState{cancel_flag, terminal_flag},
      callback_handle,
      InferenceServiceImpl::AsyncTerminalCompletionDetails{
          .service = nullptr,
          .resolved_model_name = "model",
          .request = nullptr,
          .recv_tp = MonotonicClock::now(),
          .stage = "enqueue",
          .failure_info = &failure_info,
      });
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
  return InferenceServiceImpl::AsyncOps::handle_input_validation_failure(
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
  InferenceServiceImpl::AsyncOps::finalize_successful_completion(
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
  return InferenceServiceImpl::AsyncOps::handle_job_failure(
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
  return InferenceServiceImpl::AsyncOps::prepare_async_completion(
      context, callback_handle);
}

auto
InferenceServiceTestAccessor::TryMarkTerminalNullFlagForTest() -> bool
{
  const std::shared_ptr<std::atomic<bool>> terminal_flag;
  return InferenceServiceImpl::AsyncOps::try_mark_terminal(terminal_flag);
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
  InferenceServiceImpl::AsyncOps::handle_async_infer_completion(
      context, grpc::Status::OK, outputs, breakdown, timing_info);
  return called;
}

}  // namespace starpu_server::testing
