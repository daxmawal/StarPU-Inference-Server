#pragma once

#include "inference_service.hpp"

namespace starpu_server::inference_service_runtime_internal {

auto normalize_names_runtime(
    std::vector<std::string> names, std::size_t expected_size,
    std::string_view fallback_prefix,
    std::string_view kind) -> std::vector<std::string>;
auto check_missing_named_inputs_runtime(
    const std::vector<bool>& filled,
    std::span<const std::string> expected_names) -> grpc::Status;
void arm_rpc_done_tag_with_null_context_runtime();
auto rpc_done_tag_proceed_runtime(bool is_ok, bool with_on_done) -> bool;
void set_grpc_health_status_runtime(grpc::Server* server, bool serving);
auto scalar_type_to_model_dtype_runtime(at::ScalarType type)
    -> inference::DataType;
auto resolve_tensor_name_runtime(
    std::size_t index, std::span<const std::string> names,
    std::string_view fallback_prefix) -> std::string;
auto request_batch_size_runtime(
    const inference::ModelInferRequest* request,
    int max_batch_size) -> uint64_t;
auto duration_ms_to_ns_runtime(double duration_ms) -> uint64_t;
auto elapsed_since_runtime(MonotonicClock::time_point start) -> uint64_t;
auto resolve_terminal_failure_stage_runtime(
    const grpc::Status& status, std::string_view default_stage,
    std::string_view default_reason,
    const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info)
    -> std::string;
auto should_report_terminal_failure_metric_runtime(
    const grpc::Status& status, std::string_view default_stage,
    std::string_view default_reason,
    const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info)
    -> bool;
auto is_context_cancelled_runtime(grpc::ServerContext* context) -> bool;
auto fill_output_tensor_runtime(
    inference::ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs,
    const std::vector<std::size_t>& output_indices,
    const std::vector<std::string>& output_names) -> grpc::Status;
auto validate_configured_shape_runtime(
    const std::vector<int64_t>& shape, const std::vector<int64_t>& expected,
    bool batching_allowed, int max_batch_size) -> grpc::Status;
auto unary_call_data_missing_handler_transitions_to_finish_runtime() -> bool;

}  // namespace starpu_server::inference_service_runtime_internal
