#if !defined(STARPU_TESTING)
#error \
    "inference_service_test_internal.cpp is test-only and requires STARPU_TESTING"
#endif

#include "inference_service_test_internal.hpp"

#include <future>
#include <utility>

#include "core/inference_runner.hpp"
#include "inference_service_runtime_internal.hpp"

namespace starpu_server {

auto
InferenceServiceImpl::submit_job_and_wait(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs, LatencyBreakdown& breakdown,
    detail::TimingInfo& timing_info,
    std::vector<std::shared_ptr<const void>> input_lifetimes) -> grpc::Status
{
  struct JobResult {
    grpc::Status status = grpc::Status::OK;
    std::vector<torch::Tensor> outputs;
    LatencyBreakdown breakdown;
    detail::TimingInfo timing_info;
  };

  auto result_promise = std::make_shared<std::promise<JobResult>>();
  auto result_future = result_promise->get_future();

  const auto receive_time = MonotonicClock::now();
  if (grpc::Status submit_status = submit_job_async(
          inputs,
          [result_promise](
              grpc::Status status, std::vector<torch::Tensor> outs,
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
  return grpc::Status::OK;
}

namespace testing::inference_service_test_internal {

namespace detail {

auto
handle_model_infer_async_test_hooks_ref() -> HandleModelInferAsyncTestHooks&
{
  static HandleModelInferAsyncTestHooks hooks{};
  return hooks;
}

auto
handle_async_infer_completion_test_hooks_ref()
    -> HandleAsyncInferCompletionTestHooks&
{
  static HandleAsyncInferCompletionTestHooks hooks{};
  return hooks;
}

auto
submit_job_async_test_hooks_ref() -> SubmitJobAsyncTestHooks&
{
  static SubmitJobAsyncTestHooks hooks{};
  return hooks;
}

auto
check_missing_named_inputs_override_ref() -> CheckMissingNamedInputsOverrideFn&
{
  static CheckMissingNamedInputsOverrideFn hook{};
  return hook;
}

auto
model_statistics_force_null_target_flag_ref() -> bool&
{
  static bool enabled = false;
  return enabled;
}

}  // namespace detail

void
set_handle_model_infer_async_test_hooks(HandleModelInferAsyncTestHooks hooks)
{
  detail::handle_model_infer_async_test_hooks_ref() = std::move(hooks);
}

void
clear_handle_model_infer_async_test_hooks()
{
  detail::handle_model_infer_async_test_hooks_ref() =
      HandleModelInferAsyncTestHooks{};
}

void
set_handle_async_infer_completion_test_hooks(
    HandleAsyncInferCompletionTestHooks hooks)
{
  detail::handle_async_infer_completion_test_hooks_ref() = std::move(hooks);
}

void
clear_handle_async_infer_completion_test_hooks()
{
  detail::handle_async_infer_completion_test_hooks_ref() =
      HandleAsyncInferCompletionTestHooks{};
}

void
set_submit_job_async_test_hooks(SubmitJobAsyncTestHooks hooks)
{
  detail::submit_job_async_test_hooks_ref() = std::move(hooks);
}

void
clear_submit_job_async_test_hooks()
{
  detail::submit_job_async_test_hooks_ref() = SubmitJobAsyncTestHooks{};
}

void
set_check_missing_named_inputs_override(
    detail::CheckMissingNamedInputsOverrideFn fn)
{
  detail::check_missing_named_inputs_override_ref() = std::move(fn);
}

void
clear_check_missing_named_inputs_override()
{
  detail::check_missing_named_inputs_override_ref() = {};
}

void
set_model_statistics_force_null_target_for_test(bool enable)
{
  detail::model_statistics_force_null_target_flag_ref() = enable;
}

#define STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET(                      \
    return_type, accessor_name, internal_name, runtime_name, params_decl, \
    args)                                                                 \
  auto internal_name params_decl->return_type                             \
  {                                                                       \
    return inference_service_runtime_internal::runtime_name args;         \
  }

#define STARPU_INFERENCE_SERVICE_TEST_FORWARDER_VOID(              \
    accessor_name, internal_name, runtime_name, params_decl, args) \
  void internal_name params_decl                                   \
  {                                                                \
    inference_service_runtime_internal::runtime_name args;         \
  }

#include "inference_service_test_forwarders.hpp"

#undef STARPU_INFERENCE_SERVICE_TEST_FORWARDER_VOID
#undef STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET

}  // namespace testing::inference_service_test_internal

}  // namespace starpu_server
