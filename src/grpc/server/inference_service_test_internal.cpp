#if !defined(STARPU_TESTING)
#error \
    "inference_service_test_internal.cpp is test-only and requires STARPU_TESTING"
#endif

#include "inference_service_test_internal.hpp"

#include <future>

#include "core/inference_runner.hpp"

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

auto
normalize_names_for_test(
    std::vector<std::string> names, std::size_t expected_size,
    std::string_view fallback_prefix,
    std::string_view kind) -> std::vector<std::string>
{
  return detail::normalize_names_bridge(
      std::move(names), expected_size, fallback_prefix, kind);
}

auto
check_missing_named_inputs_for_test(
    const std::vector<bool>& filled,
    std::span<const std::string> expected_names) -> grpc::Status
{
  return detail::check_missing_named_inputs_bridge(filled, expected_names);
}

void
arm_rpc_done_tag_with_null_context_for_test()
{
  detail::arm_rpc_done_tag_with_null_context_bridge();
}

auto
rpc_done_tag_proceed_for_test(bool is_ok, bool with_on_done) -> bool
{
  return detail::rpc_done_tag_proceed_bridge(is_ok, with_on_done);
}

void
set_grpc_health_status_for_test(grpc::Server* server, bool serving)
{
  detail::set_grpc_health_status_bridge(server, serving);
}

auto
scalar_type_to_model_dtype_for_test(at::ScalarType type) -> inference::DataType
{
  return detail::scalar_type_to_model_dtype_bridge(type);
}

auto
resolve_tensor_name_for_test(
    std::size_t index, std::span<const std::string> names,
    std::string_view fallback_prefix) -> std::string
{
  return detail::resolve_tensor_name_bridge(index, names, fallback_prefix);
}

auto
request_batch_size_for_test(
    const inference::ModelInferRequest* request, int max_batch_size) -> uint64_t
{
  return detail::request_batch_size_bridge(request, max_batch_size);
}

auto
duration_ms_to_ns_for_test(double duration_ms) -> uint64_t
{
  return detail::duration_ms_to_ns_bridge(duration_ms);
}

auto
elapsed_since_for_test(MonotonicClock::time_point start) -> uint64_t
{
  return detail::elapsed_since_bridge(start);
}

auto
resolve_terminal_failure_stage_for_test(
    const grpc::Status& status, std::string_view default_stage,
    std::string_view default_reason,
    const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info)
    -> std::string
{
  return detail::resolve_terminal_failure_stage_bridge(
      status, default_stage, default_reason, failure_info);
}

auto
should_report_terminal_failure_metric_for_test(
    const grpc::Status& status, std::string_view default_stage,
    std::string_view default_reason,
    const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info)
    -> bool
{
  return detail::should_report_terminal_failure_metric_bridge(
      status, default_stage, default_reason, failure_info);
}

void
set_model_statistics_force_null_target_for_test(bool enable)
{
  detail::model_statistics_force_null_target_flag_ref() = enable;
}

auto
is_context_cancelled_for_test(grpc::ServerContext* context) -> bool
{
  return detail::is_context_cancelled_bridge(context);
}

auto
fill_output_tensor_for_test(
    inference::ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs,
    const std::vector<std::size_t>& output_indices,
    const std::vector<std::string>& output_names) -> grpc::Status
{
  return detail::fill_output_tensor_bridge(
      reply, outputs, output_indices, output_names);
}

auto
validate_configured_shape_for_test(
    const std::vector<int64_t>& shape, const std::vector<int64_t>& expected,
    bool batching_allowed, int max_batch_size) -> grpc::Status
{
  return detail::validate_configured_shape_bridge(
      shape, expected, batching_allowed, max_batch_size);
}

auto
unary_call_data_missing_handler_transitions_to_finish_for_test() -> bool
{
  return detail::unary_call_data_missing_handler_transitions_to_finish_bridge();
}

}  // namespace testing::inference_service_test_internal

}  // namespace starpu_server
