#include "inference_service.hpp"

#include <grpcpp/health_check_service_interface.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#if defined(STARPU_TESTING)
#include "support/grpc/server/inference_service_test_internal.hpp"
#endif
#if defined(STARPU_ENABLE_GRPC_REFLECTION)
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#endif

#include "core/inference_runner.hpp"
#include "inference_service_runtime_internal.hpp"
#include "monitoring/congestion_monitor.hpp"
#include "monitoring/metrics.hpp"
#include "monitoring/runtime_observability.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/client_utils.hpp"
#include "utils/datatype_utils.hpp"
#include "utils/logger.hpp"
#include "utils/nvtx.hpp"

namespace starpu_server {  // NOSONAR: this translation unit includes
                           // implementation fragments.
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using inference::ModelInferRequest;
using inference::ModelInferResponse;
using inference::ModelReadyRequest;
using inference::ModelReadyResponse;
using inference::ServerLiveRequest;
using inference::ServerLiveResponse;
using inference::ServerReadyRequest;
using inference::ServerReadyResponse;

inline namespace inference_service_detail {

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto unary_call_data_missing_handler_transitions_to_finish_for_test_impl()
    -> bool;
#endif  // SONAR_IGNORE_END
auto is_context_cancelled(const ServerContext* context) -> bool;

struct NormalizeNamesOptions {
  std::string_view fallback_prefix;
  std::string_view kind;
};

auto
status_reason(const Status& status) -> std::string
{
  return std::to_string(static_cast<int>(status.error_code()));
}

struct TerminalFailureMapping {
  std::string stage;
  std::string reason;
  bool report_failure_metric = true;
};

auto
resolve_terminal_failure_mapping(
    const Status& status, std::string_view default_stage,
    std::string_view default_reason,
    const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info)
    -> TerminalFailureMapping
{
  TerminalFailureMapping mapping{};
  if (failure_info && failure_info->metrics_reported) {
    mapping.report_failure_metric = false;
    return mapping;
  }

  if (failure_info && !failure_info->stage.empty()) {
    mapping.stage = failure_info->stage;
  } else if (!default_stage.empty()) {
    mapping.stage = std::string(default_stage);
  } else {
    mapping.stage = "execution";
  }

  if (failure_info && !failure_info->reason.empty()) {
    mapping.reason = failure_info->reason;
  } else if (!default_reason.empty()) {
    mapping.reason = std::string(default_reason);
  } else {
    mapping.reason = status_reason(status);
  }

  return mapping;
}

auto
unimplemented_rpc_status(std::string_view rpc_name) -> Status
{
  return {
      grpc::StatusCode::UNIMPLEMENTED,
      std::format("RPC {} is not implemented", rpc_name)};
}

auto
normalize_names(
    std::vector<std::string> names, std::size_t expected_size,
    NormalizeNamesOptions options) -> std::vector<std::string>
{
  if (names.empty()) {
    return names;
  }

  if (const bool any_named = std::ranges::any_of(
          names, [](const auto& name) { return !name.empty(); });
      !any_named) {
    return {};
  }

  if (names.size() != expected_size || expected_size == 0) {
    log_warning(std::format(
        "Configured {} names count ({}) does not match expected count ({}); "
        "ignoring names.",
        options.kind, names.size(), expected_size));
    return {};
  }

  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i].empty()) {
      names[i] = std::format("{}{}", options.fallback_prefix, i);
    }
  }

  return names;
}

auto
check_missing_named_inputs(
    const std::vector<bool>& filled,
    std::span<const std::string> expected_names) -> Status
{
#if defined(STARPU_TESTING)
  auto& override_fn = starpu_server::testing::inference_service_test_internal::
      detail::check_missing_named_inputs_override_ref();
  if (override_fn) {
    if (auto status = override_fn(filled, expected_names); status.has_value()) {
      return *status;
    }
  }
#endif

  for (std::size_t i = 0; i < filled.size(); ++i) {
    if (!filled[i]) {
      return {
          grpc::StatusCode::INVALID_ARGUMENT,
          std::format("Missing input tensor '{}'", expected_names[i])};
    }
  }
  return Status::OK;
}

auto
scalar_type_to_model_dtype(at::ScalarType type) -> inference::DataType
{
  switch (type) {
    case at::kBool:
      return inference::DataType::TYPE_BOOL;
    case at::kByte:
      return inference::DataType::TYPE_UINT8;
    case at::kChar:
      return inference::DataType::TYPE_INT8;
    case at::kShort:
      return inference::DataType::TYPE_INT16;
    case at::kInt:
      return inference::DataType::TYPE_INT32;
    case at::kLong:
      return inference::DataType::TYPE_INT64;
    case at::kHalf:
      return inference::DataType::TYPE_FP16;
    case at::kFloat:
      return inference::DataType::TYPE_FP32;
    case at::kDouble:
      return inference::DataType::TYPE_FP64;
    case at::kBFloat16:
      return inference::DataType::TYPE_BF16;
    default:
      return inference::DataType::TYPE_INVALID;
  }
}

auto
resolve_tensor_name(
    std::size_t index, std::span<const std::string> names,
    std::string_view fallback_prefix) -> std::string
{
  if (index < names.size() && !names[index].empty()) {
    return names[index];
  }
  return std::format("{}{}", fallback_prefix, index);
}

auto
request_batch_size(const ModelInferRequest* request, int max_batch_size)
    -> uint64_t
{
  if (request == nullptr || request->inputs_size() == 0 ||
      max_batch_size <= 0) {
    return 1U;
  }
  const auto& input = request->inputs(0);
  if (input.shape_size() == 0) {
    return 1U;
  }
  const int64_t batch = input.shape(0);
  if (batch <= 0) {
    return 1U;
  }
  return static_cast<uint64_t>(batch);
}

auto
validate_model_infer_io(
    const ModelInferRequest* request, const ModelInferResponse* reply) -> Status
{
  if (request == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "ModelInfer request is null"};
  }
  if (reply == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "ModelInfer response is null"};
  }
  return Status::OK;
}

auto
duration_ms_to_ns(double duration_ms) -> uint64_t
{
  if (duration_ms <= 0.0) {
    return 0U;
  }
  constexpr double kNsPerMs = 1'000'000.0;
  const double duration_ns = duration_ms * kNsPerMs;
  if (duration_ns >=
      static_cast<double>(std::numeric_limits<uint64_t>::max())) {
    return std::numeric_limits<uint64_t>::max();
  }
  return static_cast<uint64_t>(duration_ns);
}

auto
elapsed_since(const MonotonicClock::time_point start) -> uint64_t
{
  const auto now = MonotonicClock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
  if (elapsed <= 0) {
    return 0U;
  }
  return static_cast<uint64_t>(elapsed);
}

}  // namespace inference_service_detail

inline namespace inference_service_detail {

class AsyncCallDataBase {
 public:
  explicit AsyncCallDataBase() = default;
  AsyncCallDataBase(const AsyncCallDataBase&) = delete;
  auto operator=(const AsyncCallDataBase&) -> AsyncCallDataBase& = delete;
  AsyncCallDataBase(AsyncCallDataBase&&) = default;
  auto operator=(AsyncCallDataBase&&) -> AsyncCallDataBase& = default;
  virtual ~AsyncCallDataBase() = default;
  virtual void Proceed(bool is_ok) = 0;
};

class RpcDoneTag final : public AsyncCallDataBase,
                         public std::enable_shared_from_this<RpcDoneTag> {
 public:
  using OnDone = std::function<void()>;

  static auto Create(OnDone on_done, std::shared_ptr<void> call_guard)
      -> std::shared_ptr<RpcDoneTag>
  {
    return std::make_shared<RpcDoneTag>(
        std::move(on_done), std::move(call_guard));
  }

  void Arm(grpc::ServerContext* context)
  {
    if (context == nullptr) {
      return;
    }
    self_ref_ = this->shared_from_this();
    context->AsyncNotifyWhenDone(this);
  }

  void Proceed(bool is_ok) override
  {
    if (is_ok && on_done_) {
      on_done_();
    }
    on_done_ = {};
    call_guard_.reset();
    self_ref_.reset();
  }

  RpcDoneTag(OnDone on_done, std::shared_ptr<void> call_guard)
      : on_done_(std::move(on_done)), call_guard_(std::move(call_guard))
  {
  }

 private:
  OnDone on_done_;
  std::shared_ptr<void> call_guard_;
  std::shared_ptr<RpcDoneTag> self_ref_;
};

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
#endif                       // SONAR_IGNORE_END
// GCOVR_EXCL_STOP
}  // namespace inference_service_detail


auto
compute_thread_count_from(unsigned concurrency) -> std::size_t
{
  if (concurrency == 0U) {
    return kDefaultGrpcThreads;
  }
  return std::clamp<std::size_t>(concurrency, kMinGrpcThreads, kMaxGrpcThreads);
}

inline namespace inference_service_detail {

// Input/output tensor parsing and validation helpers.
#include "inference_service_validation_convert_helpers.inc"
// Async submission/completion utility helpers.
#include "inference_service_async_coordination_helpers.inc"
}  // namespace inference_service_detail

InferenceServiceImpl::InferenceServiceImpl(
    InferenceQueue* queue, const std::vector<torch::Tensor>* reference_outputs,
    std::vector<at::ScalarType> expected_input_types,
    InputShapeConfig input_shape_config, ServiceOptions service_options,
    std::shared_ptr<RuntimeObservability> observability)
    : queue_(queue), reference_outputs_(reference_outputs),
      expected_input_types_(std::move(expected_input_types)),
      expected_input_dims_(std::move(input_shape_config.expected_input_dims)),
      max_batch_size_(input_shape_config.max_batch_size),
      default_model_name_(std::move(service_options.default_model_name)),
      server_name_(std::move(service_options.server_name)),
      server_version_(std::move(service_options.server_version)),
      observability_(std::move(observability))
{
  expected_input_names_ = normalize_names(
      std::move(service_options.expected_input_names),
      expected_input_types_.size(), NormalizeNamesOptions{"input", "input"});
  const std::size_t output_count =
      reference_outputs_ != nullptr ? reference_outputs_->size() : 0U;
  expected_output_names_ = normalize_names(
      std::move(service_options.expected_output_names), output_count,
      NormalizeNamesOptions{"output", "output"});
  validate_schema_or_throw();
}

InferenceServiceImpl::InferenceServiceImpl(
    InferenceQueue* queue, const std::vector<torch::Tensor>* reference_outputs,
    std::vector<at::ScalarType> expected_input_types,
    ServiceOptions service_options,
    std::shared_ptr<RuntimeObservability> observability)
    : InferenceServiceImpl(
          queue, reference_outputs, std::move(expected_input_types),
          InputShapeConfig{}, std::move(service_options),
          std::move(observability))
{
}

auto
InferenceServiceImpl::metrics_recorder() const
    -> std::shared_ptr<MetricsRecorder>
{
  return observability_ != nullptr ? observability_->metrics : nullptr;
}

void
InferenceServiceImpl::record_terminal_metrics(
    std::string_view model_name, const Status& status,
    std::string_view default_stage, std::string_view default_reason,
    const std::optional<AsyncFailureInfo>& failure_info,
    bool record_status_metric) const
{
  if (record_status_metric) {
    if (auto metrics = metrics_recorder(); metrics != nullptr) {
      metrics->increment_request_status(
          static_cast<int>(status.error_code()), model_name);
    } else {
      increment_request_status(
          static_cast<int>(status.error_code()), model_name);
    }
  }
  if (status.ok()) {
    return;
  }

  const auto mapping = resolve_terminal_failure_mapping(
      status, default_stage, default_reason, failure_info);
  if (!mapping.report_failure_metric) {
    return;
  }
  if (auto metrics = metrics_recorder(); metrics != nullptr) {
    metrics->increment_inference_failure(
        mapping.stage, mapping.reason, model_name);
  } else {
    increment_inference_failure(mapping.stage, mapping.reason, model_name);
  }
}

// Validation and response conversion block.
#include "inference_service_validation_convert.inc"

// Readiness/metadata/config/statistics RPC handlers.
#include "inference_service_rpc_metadata.inc"

// Async inference coordination block.
#include "inference_service_async_coordination.inc"

inline namespace inference_service_detail {
void
set_grpc_health_status(const Server* server, bool serving)
{
  if (server == nullptr) {
    return;
  }
  auto* health_service = server->GetHealthCheckService();
  if (health_service == nullptr) {
    return;
  }
  health_service->SetServingStatus(serving);
}

auto
is_context_cancelled(const ServerContext* context) -> bool
{
  if (context == nullptr) {
    return false;
  }
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  auto& test_hooks = testing::inference_service_test_internal::detail::
      handle_model_infer_async_test_hooks_ref();
  if (test_hooks.is_cancelled_override) {
    if (auto decision = test_hooks.is_cancelled_override(context);
        decision.has_value()) {
      return *decision;
    }
  }
#endif  // SONAR_IGNORE_END
  return context->IsCancelled();
}
}  // namespace inference_service_detail

namespace inference_service_runtime_internal {

auto
normalize_names_runtime(
    std::vector<std::string> names, std::size_t expected_size,
    std::string_view fallback_prefix,
    std::string_view kind) -> std::vector<std::string>
{
  return normalize_names(
      std::move(names), expected_size,
      NormalizeNamesOptions{fallback_prefix, kind});
}

auto
check_missing_named_inputs_runtime(
    const std::vector<bool>& filled,
    std::span<const std::string> expected_names) -> grpc::Status
{
  return check_missing_named_inputs(filled, expected_names);
}

void
arm_rpc_done_tag_with_null_context_runtime()
{
  auto tag = RpcDoneTag::Create(
      [] {
        // Intentionally empty: this helper only verifies Arm(nullptr).
      },
      std::make_shared<int>(0));
  tag->Arm(nullptr);
}

auto
rpc_done_tag_proceed_runtime(bool is_ok, bool with_on_done) -> bool
{
  bool called = false;
  RpcDoneTag::OnDone on_done =
      with_on_done ? [&called]() { called = true; } : RpcDoneTag::OnDone{};
  auto tag = RpcDoneTag::Create(std::move(on_done), std::make_shared<int>(0));
  tag->Proceed(is_ok);
  return called;
}

void
set_grpc_health_status_runtime(const grpc::Server* server, bool serving)
{
  set_grpc_health_status(server, serving);
}

auto
scalar_type_to_model_dtype_runtime(at::ScalarType type) -> inference::DataType
{
  return scalar_type_to_model_dtype(type);
}

auto
resolve_tensor_name_runtime(
    std::size_t index, std::span<const std::string> names,
    std::string_view fallback_prefix) -> std::string
{
  return resolve_tensor_name(index, names, fallback_prefix);
}

auto
request_batch_size_runtime(
    const inference::ModelInferRequest* request, int max_batch_size) -> uint64_t
{
  return request_batch_size(request, max_batch_size);
}

auto
duration_ms_to_ns_runtime(double duration_ms) -> uint64_t
{
  return duration_ms_to_ns(duration_ms);
}

auto
elapsed_since_runtime(MonotonicClock::time_point start) -> uint64_t
{
  return elapsed_since(start);
}

auto
resolve_terminal_failure_stage_runtime(
    const grpc::Status& status, std::string_view default_stage,
    std::string_view default_reason,
    const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info)
    -> std::string
{
  return resolve_terminal_failure_mapping(
             status, default_stage, default_reason, failure_info)
      .stage;
}

auto
should_report_terminal_failure_metric_runtime(
    const grpc::Status& status, std::string_view default_stage,
    std::string_view default_reason,
    const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info)
    -> bool
{
  return resolve_terminal_failure_mapping(
             status, default_stage, default_reason, failure_info)
      .report_failure_metric;
}

auto
is_context_cancelled_runtime(const grpc::ServerContext* context) -> bool
{
  return is_context_cancelled(context);
}

auto
fill_output_tensor_runtime(
    inference::ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs,
    const std::vector<std::size_t>& output_indices,
    const std::vector<std::string>& output_names) -> grpc::Status
{
  return fill_output_tensor(reply, outputs, output_indices, output_names);
}

auto
validate_configured_shape_runtime(
    const std::vector<int64_t>& shape, const std::vector<int64_t>& expected,
    bool batching_allowed, int max_batch_size) -> grpc::Status
{
  return validate_configured_shape(
      shape, expected, batching_allowed, max_batch_size);
}

auto
unary_call_data_missing_handler_transitions_to_finish_runtime() -> bool
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  return unary_call_data_missing_handler_transitions_to_finish_for_test_impl();
#else
  return false;
#endif  // SONAR_IGNORE_END
}

}  // namespace inference_service_runtime_internal
#include "inference_service_async_server.inc"
}  // namespace starpu_server
