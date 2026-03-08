// return_type, accessor_name, internal_name, runtime_name, params_decl, args
STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET(
    std::vector<std::string>, NormalizeNamesForTest, normalize_names_for_test,
    normalize_names_runtime,
    (std::vector<std::string> names, std::size_t expected_size,
     std::string_view fallback_prefix, std::string_view kind),
    (std::move(names), expected_size, fallback_prefix, kind))
STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET(
    grpc::Status, CheckMissingInputsForTest, check_missing_named_inputs_for_test,
    check_missing_named_inputs_runtime,
    (const std::vector<bool>& filled, std::span<const std::string> expected_names),
    (filled, expected_names))
STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET(
    bool, RpcDoneTagProceedForTest, rpc_done_tag_proceed_for_test,
    rpc_done_tag_proceed_runtime, (bool is_ok, bool with_on_done),
    (is_ok, with_on_done))
STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET(
    inference::DataType, ScalarTypeToModelDtypeForTest,
    scalar_type_to_model_dtype_for_test, scalar_type_to_model_dtype_runtime,
    (at::ScalarType type), (type))
STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET(
    std::string, ResolveTensorNameForTest, resolve_tensor_name_for_test,
    resolve_tensor_name_runtime,
    (std::size_t index, std::span<const std::string> names,
     std::string_view fallback_prefix),
    (index, names, fallback_prefix))
STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET(
    uint64_t, RequestBatchSizeForTest, request_batch_size_for_test,
    request_batch_size_runtime,
    (const inference::ModelInferRequest* request, int max_batch_size),
    (request, max_batch_size))
STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET(
    uint64_t, DurationMsToNsForTest, duration_ms_to_ns_for_test,
    duration_ms_to_ns_runtime, (double duration_ms), (duration_ms))
STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET(
    uint64_t, ElapsedSinceForTest, elapsed_since_for_test, elapsed_since_runtime,
    (MonotonicClock::time_point start), (start))
STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET(
    std::string, ResolveTerminalFailureStageForTest,
    resolve_terminal_failure_stage_for_test, resolve_terminal_failure_stage_runtime,
    (const grpc::Status& status, std::string_view default_stage,
     std::string_view default_reason,
     const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info),
    (status, default_stage, default_reason, failure_info))
STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET(
    bool, ShouldReportTerminalFailureMetricForTest,
    should_report_terminal_failure_metric_for_test,
    should_report_terminal_failure_metric_runtime,
    (const grpc::Status& status, std::string_view default_stage,
     std::string_view default_reason,
     const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info),
    (status, default_stage, default_reason, failure_info))
STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET(
    bool, IsContextCancelledForTest, is_context_cancelled_for_test,
    is_context_cancelled_runtime, (grpc::ServerContext* context), (context))
STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET(
    grpc::Status, FillOutputTensorForTest, fill_output_tensor_for_test,
    fill_output_tensor_runtime,
    (inference::ModelInferResponse* reply,
     const std::vector<torch::Tensor>& outputs,
     const std::vector<std::size_t>& output_indices,
     const std::vector<std::string>& output_names),
    (reply, outputs, output_indices, output_names))
STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET(
    grpc::Status, ValidateConfiguredShapeForTest, validate_configured_shape_for_test,
    validate_configured_shape_runtime,
    (const std::vector<int64_t>& shape, const std::vector<int64_t>& expected,
     bool batching_allowed, int max_batch_size),
    (shape, expected, batching_allowed, max_batch_size))
STARPU_INFERENCE_SERVICE_TEST_FORWARDER_RET(
    bool, UnaryCallDataMissingHandlerTransitionsToFinishForTest,
    unary_call_data_missing_handler_transitions_to_finish_for_test,
    unary_call_data_missing_handler_transitions_to_finish_runtime, (), ())

// accessor_name, internal_name, runtime_name, params_decl, args
STARPU_INFERENCE_SERVICE_TEST_FORWARDER_VOID(
    ArmRpcDoneTagWithNullContextForTest, arm_rpc_done_tag_with_null_context_for_test,
    arm_rpc_done_tag_with_null_context_runtime, (), ())
STARPU_INFERENCE_SERVICE_TEST_FORWARDER_VOID(
    SetGrpcHealthStatusForTest, set_grpc_health_status_for_test,
    set_grpc_health_status_runtime, (grpc::Server* server, bool serving),
    (server, serving))
