#pragma once

#if !defined(STARPU_TESTING)
#error "inference_service_test_api.hpp is test-only and requires STARPU_TESTING"
#endif

#include "inference_service.hpp"

namespace starpu_server::testing {

auto ModelInferForTest(
    InferenceServiceImpl& service, grpc::ServerContext* context,
    const inference::ModelInferRequest* request,
    inference::ModelInferResponse* reply) -> grpc::Status;

auto SubmitJobAndWaitForTest(
    InferenceServiceImpl& service, const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    InferenceServiceImpl::LatencyBreakdown& breakdown,
    detail::TimingInfo& timing_info,
    std::vector<std::shared_ptr<const void>> input_lifetimes = {})
    -> grpc::Status;

struct HandleModelInferAsyncTestHooks {
  std::function<void(const std::shared_ptr<std::atomic<bool>>&)>
      on_cancel_flag_created;
  std::function<void(const std::function<void()>&)> on_cancel_ready;
  std::function<std::optional<bool>(const grpc::ServerContext*)>
      is_cancelled_override;
  std::function<void(
      const std::shared_ptr<std::atomic<bool>>&, const grpc::Status&)>
      on_submit_job_async_done;
};

struct HandleAsyncInferCompletionTestHooks {
  std::function<void(const std::shared_ptr<std::atomic<bool>>&)>
      after_try_acquire;
  std::function<void(const std::shared_ptr<std::atomic<bool>>&)>
      before_final_cancel_check;
  std::function<void(const std::shared_ptr<std::atomic<bool>>&)>
      before_success_terminal_mark;
};

struct SubmitJobAsyncTestHooks {
  std::function<void()> before_create_job;
};

using CheckMissingNamedInputsOverrideFn =
    std::function<std::optional<grpc::Status>(
        const std::vector<bool>&, std::span<const std::string>)>;

class InferenceServiceTestAccessor {
 public:
  static void SetHandleModelInferAsyncTestHooks(
      HandleModelInferAsyncTestHooks hooks);
  static void ClearHandleModelInferAsyncTestHooks();
  static void SetHandleAsyncInferCompletionTestHooks(
      HandleAsyncInferCompletionTestHooks hooks);
  static void ClearHandleAsyncInferCompletionTestHooks();
  static void SetSubmitJobAsyncTestHooks(SubmitJobAsyncTestHooks hooks);
  static void ClearSubmitJobAsyncTestHooks();
  static void SetCheckMissingNamedInputsOverrideForTest(
      CheckMissingNamedInputsOverrideFn override_fn);
  static void ClearCheckMissingNamedInputsOverrideForTest();

  static auto NormalizeNamesForTest(
      std::vector<std::string> names, std::size_t expected_size,
      std::string_view fallback_prefix,
      std::string_view kind) -> std::vector<std::string>;
  static void SetExpectedInputTypesForTest(
      InferenceServiceImpl* service, std::vector<at::ScalarType> types);
  static void SetExpectedInputNamesForTest(
      InferenceServiceImpl* service, std::vector<std::string> names);
  static void SetReferenceOutputsForTest(
      InferenceServiceImpl* service,
      const std::vector<torch::Tensor>* reference_outputs);
  static auto CheckMissingInputsForTest(
      const std::vector<bool>& filled,
      std::span<const std::string> expected_names) -> grpc::Status;
  static void ArmRpcDoneTagWithNullContextForTest();
  static auto RpcDoneTagProceedForTest(bool is_ok, bool with_on_done) -> bool;
  static void SetGrpcHealthStatusForTest(grpc::Server* server, bool serving);
  static void RecordSuccessForTest(
      InferenceServiceImpl* service,
      const inference::ModelInferRequest* request,
      const InferenceServiceImpl::LatencyBreakdown& breakdown,
      MonotonicClock::time_point recv_tp, std::string_view resolved_model_name);
  static void RecordFailureForTest(
      InferenceServiceImpl* service,
      const inference::ModelInferRequest* request,
      MonotonicClock::time_point recv_tp, std::string_view resolved_model_name);
  static auto ScalarTypeToModelDtypeForTest(at::ScalarType type)
      -> inference::DataType;
  static auto ResolveTensorNameForTest(
      std::size_t index, std::span<const std::string> names,
      std::string_view fallback_prefix) -> std::string;
  static auto RequestBatchSizeForTest(
      const inference::ModelInferRequest* request,
      int max_batch_size) -> uint64_t;
  static auto DurationMsToNsForTest(double duration_ms) -> uint64_t;
  static auto ElapsedSinceForTest(MonotonicClock::time_point start) -> uint64_t;
  static auto ResolveTerminalFailureStageForTest(
      const grpc::Status& status, std::string_view default_stage,
      std::string_view default_reason,
      const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info)
      -> std::string;
  static auto ShouldReportTerminalFailureMetricForTest(
      const grpc::Status& status, std::string_view default_stage,
      std::string_view default_reason,
      const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info)
      -> bool;
  static void SetModelStatisticsForceNullTargetForTest(bool enable);
  static auto IsContextCancelledForTest(grpc::ServerContext* context) -> bool;
  static auto FillOutputTensorForTest(
      inference::ModelInferResponse* reply,
      const std::vector<torch::Tensor>& outputs,
      const std::vector<std::size_t>& output_indices,
      const std::vector<std::string>& output_names) -> grpc::Status;
  static auto BuildLatencyBreakdownForTest(
      const detail::TimingInfo& info,
      double latency_ms) -> InferenceServiceImpl::LatencyBreakdown;
  static auto HandleSubmitFailureForTest(
      const grpc::Status& status, bool cancelled, bool terminal_marked,
      std::atomic<bool>* callback_invoked,
      const std::optional<InferenceServiceImpl::AsyncFailureInfo>&
          failure_info = std::nullopt) -> bool;
  static auto HandleInputValidationFailureForTest(
      const grpc::Status& status, bool cancelled, bool terminal_marked,
      std::atomic<bool>* callback_invoked) -> bool;
  static auto FinalizeSuccessfulCompletionForTest(
      bool cancelled, bool terminal_marked, bool callback_present,
      bool reply_present, bool with_impl,
      std::atomic<bool>* callback_invoked) -> bool;
  static auto HandleJobFailureForTest(
      const grpc::Status& job_status, bool terminal_marked,
      bool callback_present, std::atomic<bool>* callback_invoked) -> bool;
  static auto PrepareAsyncCompletionForTest(
      bool cancelled, bool callback_present) -> bool;
  static auto TryMarkTerminalNullFlagForTest() -> bool;
  static auto HandleAsyncInferCompletionForTest(bool cancelled) -> bool;
  static auto ValidateConfiguredShapeForTest(
      const std::vector<int64_t>& shape, const std::vector<int64_t>& expected,
      bool batching_allowed, int max_batch_size) -> grpc::Status;
  static auto UnaryCallDataMissingHandlerTransitionsToFinishForTest() -> bool;
};

}  // namespace starpu_server::testing
