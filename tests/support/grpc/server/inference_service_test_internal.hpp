#pragma once

#if !defined(STARPU_TESTING)
#error \
    "inference_service_test_internal.hpp is test-only and requires STARPU_TESTING"
#endif

#include "inference_service_test_api.hpp"

namespace starpu_server::testing::inference_service_test_internal {

namespace detail {

auto handle_model_infer_async_test_hooks_ref()
    -> HandleModelInferAsyncTestHooks&;
auto handle_async_infer_completion_test_hooks_ref()
    -> HandleAsyncInferCompletionTestHooks&;
auto submit_job_async_test_hooks_ref() -> SubmitJobAsyncTestHooks&;
using CheckMissingNamedInputsOverrideFn =
    std::function<std::optional<grpc::Status>(
        const std::vector<bool>&, std::span<const std::string>)>;
auto check_missing_named_inputs_override_ref()
    -> CheckMissingNamedInputsOverrideFn&;
auto model_statistics_force_null_target_flag_ref() -> bool&;

}  // namespace detail

void set_handle_model_infer_async_test_hooks(
    HandleModelInferAsyncTestHooks hooks);
void clear_handle_model_infer_async_test_hooks();
void set_handle_async_infer_completion_test_hooks(
    HandleAsyncInferCompletionTestHooks hooks);
void clear_handle_async_infer_completion_test_hooks();
void set_submit_job_async_test_hooks(SubmitJobAsyncTestHooks hooks);
void clear_submit_job_async_test_hooks();
void set_check_missing_named_inputs_override(
    detail::CheckMissingNamedInputsOverrideFn override_fn);
void clear_check_missing_named_inputs_override();

auto normalize_names_for_test(
    std::vector<std::string> names, std::size_t expected_size,
    std::string_view fallback_prefix,
    std::string_view kind) -> std::vector<std::string>;
auto check_missing_named_inputs_for_test(
    const std::vector<bool>& filled,
    std::span<const std::string> expected_names) -> grpc::Status;
void arm_rpc_done_tag_with_null_context_for_test();
auto rpc_done_tag_proceed_for_test(bool is_ok, bool with_on_done) -> bool;
void set_grpc_health_status_for_test(grpc::Server* server, bool serving);
auto scalar_type_to_model_dtype_for_test(at::ScalarType type)
    -> inference::DataType;
auto resolve_tensor_name_for_test(
    std::size_t index, std::span<const std::string> names,
    std::string_view fallback_prefix) -> std::string;
auto request_batch_size_for_test(
    const inference::ModelInferRequest* request,
    int max_batch_size) -> uint64_t;
auto duration_ms_to_ns_for_test(double duration_ms) -> uint64_t;
auto elapsed_since_for_test(MonotonicClock::time_point start) -> uint64_t;
auto resolve_terminal_failure_stage_for_test(
    const grpc::Status& status, std::string_view default_stage,
    std::string_view default_reason,
    const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info)
    -> std::string;
auto should_report_terminal_failure_metric_for_test(
    const grpc::Status& status, std::string_view default_stage,
    std::string_view default_reason,
    const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info)
    -> bool;
void set_model_statistics_force_null_target_for_test(bool enable);
auto is_context_cancelled_for_test(grpc::ServerContext* context) -> bool;
auto fill_output_tensor_for_test(
    inference::ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs,
    const std::vector<std::size_t>& output_indices,
    const std::vector<std::string>& output_names) -> grpc::Status;
auto validate_configured_shape_for_test(
    const std::vector<int64_t>& shape, const std::vector<int64_t>& expected,
    bool batching_allowed, int max_batch_size) -> grpc::Status;
auto unary_call_data_missing_handler_transitions_to_finish_for_test() -> bool;

}  // namespace starpu_server::testing::inference_service_test_internal
