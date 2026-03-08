#include "inference_service.hpp"

#include <grpcpp/health_check_service_interface.h>

#include <algorithm>
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
#include "inference_service_test_internal.hpp"
#endif
#if defined(STARPU_ENABLE_GRPC_REFLECTION)
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#endif

#include "core/inference_runner.hpp"
#include "monitoring/congestion_monitor.hpp"
#include "monitoring/metrics.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/client_utils.hpp"
#include "utils/datatype_utils.hpp"
#include "utils/logger.hpp"
#include "utils/nvtx.hpp"

namespace starpu_server {
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

namespace {

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto unary_call_data_missing_handler_transitions_to_finish_for_test_impl()
    -> bool;
#endif  // SONAR_IGNORE_END

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

void
record_terminal_metrics(
    std::string_view model_name, const Status& status,
    std::string_view default_stage, std::string_view default_reason = {},
    const std::optional<InferenceServiceImpl::AsyncFailureInfo>& failure_info =
        std::nullopt,
    bool record_status_metric = true)
{
  if (record_status_metric) {
    increment_request_status(static_cast<int>(status.error_code()), model_name);
  }
  if (status.ok()) {
    return;
  }

  const auto mapping = resolve_terminal_failure_mapping(
      status, default_stage, default_reason, failure_info);
  if (mapping.report_failure_metric) {
    increment_inference_failure(mapping.stage, mapping.reason, model_name);
  }
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

}  // namespace

namespace {

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
}  // namespace


auto
compute_thread_count_from(unsigned concurrency) -> std::size_t
{
  if (concurrency == 0U) {
    return kDefaultGrpcThreads;
  }
  return std::clamp<std::size_t>(concurrency, kMinGrpcThreads, kMaxGrpcThreads);
}

namespace {

// Input/output tensor parsing and validation helpers.
#include "inference_service_io_validation.inl"
// Async submission/completion utility helpers.
#include "inference_service_async_lifecycle.inl"
}  // namespace

InferenceServiceImpl::InferenceServiceImpl(
    InferenceQueue* queue, const std::vector<torch::Tensor>* reference_outputs,
    std::vector<at::ScalarType> expected_input_types,
    InputShapeConfig input_shape_config, ServiceOptions service_options)
    : queue_(queue), reference_outputs_(reference_outputs),
      expected_input_types_(std::move(expected_input_types)),
      expected_input_dims_(std::move(input_shape_config.expected_input_dims)),
      max_batch_size_(input_shape_config.max_batch_size),
      default_model_name_(std::move(service_options.default_model_name)),
      server_name_(std::move(service_options.server_name)),
      server_version_(std::move(service_options.server_version))
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
    ServiceOptions service_options)
    : InferenceServiceImpl(
          queue, reference_outputs, std::move(expected_input_types),
          InputShapeConfig{}, std::move(service_options))
{
}

// Readiness/metadata/config/statistics RPC handlers.
#include "inference_service_rpc_metadata.inl"

// Infer RPC path: request validation, async orchestration, response shaping.
#include "inference_service_rpc_infer.inl"

namespace {
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
is_context_cancelled(ServerContext* context) -> bool
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
}  // namespace

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto
starpu_server::testing::inference_service_test_internal::detail::
    handle_model_infer_async_test_hooks_ref() -> HandleModelInferAsyncTestHooks&
{
  static HandleModelInferAsyncTestHooks hooks{};
  return hooks;
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    handle_async_infer_completion_test_hooks_ref()
        -> HandleAsyncInferCompletionTestHooks&
{
  static HandleAsyncInferCompletionTestHooks hooks{};
  return hooks;
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    submit_job_async_test_hooks_ref() -> SubmitJobAsyncTestHooks&
{
  static SubmitJobAsyncTestHooks hooks{};
  return hooks;
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    model_statistics_force_null_target_flag_ref() -> bool&
{
  static bool enabled = false;
  return enabled;
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    normalize_names_bridge(
        std::vector<std::string> names, std::size_t expected_size,
        std::string_view fallback_prefix,
        std::string_view kind) -> std::vector<std::string>
{
  return normalize_names(
      std::move(names), expected_size,
      NormalizeNamesOptions{fallback_prefix, kind});
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    check_missing_named_inputs_bridge(
        const std::vector<bool>& filled,
        std::span<const std::string> expected_names) -> grpc::Status
{
  return check_missing_named_inputs(filled, expected_names);
}

void
starpu_server::testing::inference_service_test_internal::detail::
    arm_rpc_done_tag_with_null_context_bridge()
{
  auto tag = RpcDoneTag::Create([] {}, std::make_shared<int>(0));
  tag->Arm(nullptr);
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    rpc_done_tag_proceed_bridge(bool is_ok, bool with_on_done) -> bool
{
  bool called = false;
  RpcDoneTag::OnDone on_done =
      with_on_done ? [&called]() { called = true; } : RpcDoneTag::OnDone{};
  auto tag = RpcDoneTag::Create(std::move(on_done), std::make_shared<int>(0));
  tag->Proceed(is_ok);
  return called;
}

void
starpu_server::testing::inference_service_test_internal::detail::
    set_grpc_health_status_bridge(grpc::Server* server, bool serving)
{
  set_grpc_health_status(server, serving);
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    scalar_type_to_model_dtype_bridge(at::ScalarType type)
        -> inference::DataType
{
  return scalar_type_to_model_dtype(type);
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    resolve_tensor_name_bridge(
        std::size_t index, std::span<const std::string> names,
        std::string_view fallback_prefix) -> std::string
{
  return resolve_tensor_name(index, names, fallback_prefix);
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    request_batch_size_bridge(
        const ModelInferRequest* request, int max_batch_size) -> uint64_t
{
  return request_batch_size(request, max_batch_size);
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    duration_ms_to_ns_bridge(double duration_ms) -> uint64_t
{
  return duration_ms_to_ns(duration_ms);
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    elapsed_since_bridge(MonotonicClock::time_point start) -> uint64_t
{
  return elapsed_since(start);
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    resolve_terminal_failure_stage_bridge(
        const grpc::Status& status, std::string_view default_stage,
        std::string_view default_reason,
        const std::optional<InferenceServiceImpl::AsyncFailureInfo>&
            failure_info) -> std::string
{
  return resolve_terminal_failure_mapping(
             status, default_stage, default_reason, failure_info)
      .stage;
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    should_report_terminal_failure_metric_bridge(
        const grpc::Status& status, std::string_view default_stage,
        std::string_view default_reason,
        const std::optional<InferenceServiceImpl::AsyncFailureInfo>&
            failure_info) -> bool
{
  return resolve_terminal_failure_mapping(
             status, default_stage, default_reason, failure_info)
      .report_failure_metric;
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    is_context_cancelled_bridge(grpc::ServerContext* context) -> bool
{
  return is_context_cancelled(context);
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    fill_output_tensor_bridge(
        inference::ModelInferResponse* reply,
        const std::vector<torch::Tensor>& outputs,
        const std::vector<std::size_t>& output_indices,
        const std::vector<std::string>& output_names) -> grpc::Status
{
  return fill_output_tensor(reply, outputs, output_indices, output_names);
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    validate_configured_shape_bridge(
        const std::vector<int64_t>& shape, const std::vector<int64_t>& expected,
        bool batching_allowed, int max_batch_size) -> grpc::Status
{
  return validate_configured_shape(
      shape, expected, batching_allowed, max_batch_size);
}

auto
starpu_server::testing::inference_service_test_internal::detail::
    unary_call_data_missing_handler_transitions_to_finish_bridge() -> bool
{
  return unary_call_data_missing_handler_transitions_to_finish_for_test_impl();
}
#endif  // SONAR_IGNORE_END

InferenceServiceImpl::CallbackHandle::CallbackHandle(
    std::function<void(Status)> callback)
    : callback_(std::move(callback))
{
}

auto
InferenceServiceImpl::CallbackHandle::TryAcquire() -> bool
{
  std::scoped_lock lock(mutex_);
  if (consumed_) {
    return false;
  }
  consumed_ = true;
  return true;
}

auto
InferenceServiceImpl::CallbackHandle::Invoke(Status status) -> bool
{
  std::function<void(Status)> callback;
  {
    std::scoped_lock lock(mutex_);
    if (!callback_) {
      return false;
    }
    consumed_ = true;
    callback = std::move(callback_);
  }
  callback(std::move(status));
  return true;
}

auto
InferenceServiceImpl::try_mark_terminal(
    const std::shared_ptr<std::atomic<bool>>& terminal_flag) -> bool
{
  if (terminal_flag == nullptr) {
    return true;
  }
  bool expected = false;
  return terminal_flag->compare_exchange_strong(
      expected, true, std::memory_order_acq_rel, std::memory_order_acquire);
}

auto
InferenceServiceImpl::enter_async_terminal_once(
    const AsyncTerminalState& terminal_state, bool check_cancel_flag) -> bool
{
  if (check_cancel_flag && terminal_state.cancel_flag != nullptr &&
      terminal_state.cancel_flag->load(std::memory_order_acquire)) {
    return false;
  }
  return try_mark_terminal(terminal_state.terminal_flag);
}

auto
InferenceServiceImpl::invoke_async_callback(
    const std::shared_ptr<CallbackHandle>& callback_handle,
    const Status& status) -> bool
{
  return callback_handle != nullptr && callback_handle->Invoke(status);
}

void
InferenceServiceImpl::record_async_terminal_failure(
    InferenceServiceImpl* service, std::string_view resolved_model_name,
    const ModelInferRequest* request, MonotonicClock::time_point recv_tp,
    const Status& status, std::string_view default_stage,
    std::string_view default_reason,
    const std::optional<AsyncFailureInfo>& failure_info,
    bool record_status_metric)
{
  if (service != nullptr) {
    service->record_failure(request, recv_tp, resolved_model_name);
  }
  record_terminal_metrics(
      resolved_model_name, status, default_stage, default_reason, failure_info,
      record_status_metric);
}

auto
InferenceServiceImpl::is_async_cancelled(
    const AsyncInferCompletionContext& context) -> bool
{
  return context.cancel_flag != nullptr &&
         context.cancel_flag->load(std::memory_order_acquire);
}

auto
InferenceServiceImpl::prepare_async_completion(
    const AsyncInferCompletionContext& context,
    const std::shared_ptr<CallbackHandle>& callback_handle) -> bool
{
  if (is_async_cancelled(context)) {
    return false;
  }
  if (callback_handle == nullptr) {
    return false;
  }
  if (!callback_handle->TryAcquire()) {
    return false;
  }
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  auto& async_hooks = testing::inference_service_test_internal::detail::
      handle_async_infer_completion_test_hooks_ref();
  if (async_hooks.after_try_acquire && context.cancel_flag != nullptr) {
    async_hooks.after_try_acquire(context.cancel_flag);
  }
#endif  // SONAR_IGNORE_END
  if (is_async_cancelled(context)) {
    return false;
  }
  return true;
}

auto
InferenceServiceImpl::handle_job_failure(
    const AsyncInferCompletionContext& context, const Status& job_status,
    const std::shared_ptr<CallbackHandle>& callback_handle) -> bool
{
  if (job_status.ok()) {
    return false;
  }
  if (!try_mark_terminal(context.terminal_flag)) {
    return true;
  }
  record_terminal_metrics(
      context.resolved_model_name, job_status, "execution", {},
      context.failure_info);
  if (context.impl != nullptr) {
    context.impl->record_failure(
        context.request, context.recv_tp, context.resolved_model_name);
  }
  if (callback_handle != nullptr) {
    callback_handle->Invoke(job_status);
  }
  return true;
}

void
InferenceServiceImpl::finalize_successful_completion(
    const AsyncInferCompletionContext& context,
    const std::vector<torch::Tensor>& outs, LatencyBreakdown breakdown,
    const detail::TimingInfo& timing_info)
{
  const auto& callback_handle = context.callback_handle;
  if (callback_handle == nullptr) {
    if (!try_mark_terminal(context.terminal_flag)) {
      return;
    }
    if (context.impl != nullptr) {
      context.impl->record_failure(
          context.request, context.recv_tp, context.resolved_model_name);
    }
    const Status missing_callback_status{
        grpc::StatusCode::INTERNAL, "Internal server error"};
    record_terminal_metrics(
        context.resolved_model_name, missing_callback_status, "postprocess",
        "missing_callback", std::nullopt, /*record_status_metric=*/false);
    log_error("Missing callback handle during async inference completion");
    return;
  }
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  auto& async_hooks = testing::inference_service_test_internal::detail::
      handle_async_infer_completion_test_hooks_ref();
#endif  // SONAR_IGNORE_END

  const auto zero_tp = MonotonicClock::time_point{};
  if (timing_info.enqueued_time > zero_tp) {
    const auto preprocess_duration = std::chrono::duration<double, std::milli>(
        timing_info.enqueued_time - context.recv_tp);
    breakdown.preprocess_ms = std::max(0.0, preprocess_duration.count());
  } else {
    breakdown.preprocess_ms = 0.0;
  }

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (async_hooks.before_final_cancel_check && context.cancel_flag != nullptr) {
    async_hooks.before_final_cancel_check(context.cancel_flag);
  }
#endif  // SONAR_IGNORE_END
  if (is_async_cancelled(context)) {
    return;
  }

  const auto output_names =
      context.output_names != nullptr
          ? std::span<const std::string>(*context.output_names)
          : std::span<const std::string>{};
  InferenceServiceImpl::PopulateResponseOptions response_options;
  response_options.model_name_override = context.resolved_model_name;
  response_options.set_prepost_overall = false;
  response_options.output_names = output_names;
  Status populate_status = populate_response(
      context.request, context.reply, outs, context.recv_ms, breakdown,
      response_options);
  if (!populate_status.ok()) {
    if (!try_mark_terminal(context.terminal_flag)) {
      return;
    }
    if (context.impl != nullptr) {
      context.impl->record_failure(
          context.request, context.recv_tp, context.resolved_model_name);
    }
    record_terminal_metrics(
        context.resolved_model_name, populate_status, "postprocess");
    callback_handle->Invoke(populate_status);
    return;
  }

  const auto send_tp = MonotonicClock::now();
  const int64_t send_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  context.reply->set_server_send_ms(send_ms);

  if (timing_info.callback_end_time > zero_tp) {
    const auto postprocess_duration = std::chrono::duration<double, std::milli>(
        send_tp - timing_info.callback_end_time);
    breakdown.postprocess_ms = std::max(0.0, postprocess_duration.count());
  } else {
    breakdown.postprocess_ms = 0.0;
  }

  breakdown.overall_ms = std::max(
      0.0, std::chrono::duration<double, std::milli>(send_tp - context.recv_tp)
               .count());

  context.reply->set_server_preprocess_ms(breakdown.preprocess_ms);
  context.reply->set_server_postprocess_ms(breakdown.postprocess_ms);
  context.reply->set_server_overall_ms(breakdown.overall_ms);

  if (context.metrics &&
      context.metrics->histograms().inference_latency != nullptr) {
    const auto latency_ms =
        std::chrono::duration<double, std::milli>(send_tp - context.recv_tp)
            .count();
    context.metrics->histograms().inference_latency->Observe(latency_ms);
  }

  observe_latency_breakdown(LatencyBreakdownMetrics{
      breakdown.queue_ms,
      breakdown.batch_ms,
      breakdown.submit_ms,
      breakdown.scheduling_ms,
      breakdown.codelet_ms,
      breakdown.inference_ms,
      breakdown.callback_ms,
      breakdown.preprocess_ms,
      breakdown.postprocess_ms,
  });

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (async_hooks.before_success_terminal_mark &&
      context.cancel_flag != nullptr) {
    async_hooks.before_success_terminal_mark(context.cancel_flag);
  }
#endif  // SONAR_IGNORE_END
  if (is_async_cancelled(context)) {
    return;
  }
  if (!try_mark_terminal(context.terminal_flag)) {
    return;
  }

  if (context.impl != nullptr) {
    context.impl->record_success(
        context.request, breakdown, context.recv_tp,
        context.resolved_model_name);
  }
  record_terminal_metrics(context.resolved_model_name, Status::OK, "");
  callback_handle->Invoke(Status::OK);
}

void
InferenceServiceImpl::handle_async_infer_completion(
    const AsyncInferCompletionContext& context, const Status& job_status,
    const std::vector<torch::Tensor>& outs, LatencyBreakdown breakdown,
    detail::TimingInfo timing_info)
{
  const auto& callback_handle = context.callback_handle;
  if (!prepare_async_completion(context, callback_handle)) {
    return;
  }
  if (handle_job_failure(context, job_status, callback_handle)) {
    return;
  }
  finalize_successful_completion(context, outs, breakdown, timing_info);
}

void
InferenceServiceImpl::notify_cancel_flag_created(
    const std::shared_ptr<std::atomic<bool>>& cancel_flag)
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  auto& test_hooks = testing::inference_service_test_internal::detail::
      handle_model_infer_async_test_hooks_ref();
  if (test_hooks.on_cancel_flag_created) {
    test_hooks.on_cancel_flag_created(cancel_flag);
  }
#else
  (void)cancel_flag;
#endif  // SONAR_IGNORE_END
}

void
InferenceServiceImpl::notify_submit_job_async_done(
    const std::shared_ptr<std::atomic<bool>>& cancel_flag, const Status& status)
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  auto& test_hooks = testing::inference_service_test_internal::detail::
      handle_model_infer_async_test_hooks_ref();
  if (test_hooks.on_submit_job_async_done) {
    test_hooks.on_submit_job_async_done(cancel_flag, status);
  }
#else
  (void)cancel_flag;
  (void)status;
#endif  // SONAR_IGNORE_END
}

auto
InferenceServiceImpl::setup_async_cancellation(
    ServerContext* context, std::shared_ptr<void>& call_guard,
    const AsyncCancellationContext& cancellation_context) -> bool
{
  if (context == nullptr || !call_guard ||
      cancellation_context.cancel_flag == nullptr ||
      cancellation_context.callback_handle == nullptr) {
    return false;
  }

  auto on_cancel = [context, cancel_flag = cancellation_context.cancel_flag,
                    terminal_flag = cancellation_context.terminal_flag,
                    callback_handle = cancellation_context.callback_handle,
                    model_name =
                        std::string(cancellation_context.resolved_model_name),
                    service = cancellation_context.service,
                    request = cancellation_context.request,
                    recv_tp = cancellation_context.recv_tp]() {
    if (!is_context_cancelled(context)) {
      return;
    }
    if (cancel_flag->exchange(true, std::memory_order_acq_rel)) {
      return;
    }
    const AsyncTerminalState terminal_state{cancel_flag, terminal_flag};
    if (!enter_async_terminal_once(
            terminal_state, /*check_cancel_flag=*/false)) {
      return;
    }
    const Status cancelled_status{
        grpc::StatusCode::CANCELLED, "Request cancelled"};
    if (!invoke_async_callback(callback_handle, cancelled_status)) {
      return;
    }
    record_async_terminal_failure(
        service, model_name, request, recv_tp, cancelled_status, "cancel",
        "client_cancelled");
  };

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  auto& test_hooks = testing::inference_service_test_internal::detail::
      handle_model_infer_async_test_hooks_ref();
  if (test_hooks.on_cancel_ready) {
    test_hooks.on_cancel_ready(on_cancel);
  }
#endif  // SONAR_IGNORE_END

  auto done_tag = RpcDoneTag::Create(on_cancel, std::move(call_guard));
  done_tag->Arm(context);
  if (is_context_cancelled(context)) {
    on_cancel();
    return true;
  }
  return false;
}

auto
InferenceServiceImpl::handle_input_validation_failure(
    const Status& status,
    const InferenceServiceImpl::AsyncTerminalState& terminal_state,
    const std::shared_ptr<CallbackHandle>& callback_handle,
    InferenceServiceImpl* service, std::string_view resolved_model_name,
    const ModelInferRequest* request,
    MonotonicClock::time_point recv_tp) -> bool
{
  if (status.ok()) {
    return false;
  }
  if (!enter_async_terminal_once(terminal_state)) {
    return true;
  }
  record_async_terminal_failure(
      service, resolved_model_name, request, recv_tp, status, "preprocess");
  (void)invoke_async_callback(callback_handle, status);
  return true;
}

auto
InferenceServiceImpl::handle_submit_failure(
    const Status& status,
    const InferenceServiceImpl::AsyncTerminalState& terminal_state,
    const std::shared_ptr<CallbackHandle>& callback_handle,
    InferenceServiceImpl* service, std::string_view resolved_model_name,
    const ModelInferRequest* request, MonotonicClock::time_point recv_tp,
    const std::optional<AsyncFailureInfo>& failure_info) -> bool
{
  if (status.ok()) {
    return false;
  }
  if (!enter_async_terminal_once(terminal_state)) {
    return true;
  }
  record_async_terminal_failure(
      service, resolved_model_name, request, recv_tp, status, "enqueue", {},
      failure_info);
  (void)invoke_async_callback(callback_handle, status);
  return true;
}

void
InferenceServiceImpl::handle_async_internal_error(
    const InferenceServiceImpl::AsyncTerminalState& terminal_state,
    const std::shared_ptr<CallbackHandle>& callback_handle,
    InferenceServiceImpl* service, std::string_view resolved_model_name,
    const ModelInferRequest* request, MonotonicClock::time_point recv_tp,
    const InferenceServiceImpl::AsyncInternalErrorDetails& details)
{
  if (!enter_async_terminal_once(terminal_state)) {
    return;
  }

  const Status status{grpc::StatusCode::INTERNAL, "Internal server error"};
  if (!invoke_async_callback(callback_handle, status)) {
    return;
  }

  record_async_terminal_failure(
      service, resolved_model_name, request, recv_tp, status, details.stage,
      details.reason);
  log_error(std::string(details.log_context));
}

void
InferenceServiceImpl::record_success(
    const ModelInferRequest* request, const LatencyBreakdown& breakdown,
    MonotonicClock::time_point recv_tp, std::string_view resolved_model_name)
{
  if (request == nullptr) {
    return;
  }
  const auto now_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  const uint64_t batch_size = request_batch_size(request, max_batch_size_);
  const uint64_t total_ns = duration_ms_to_ns(
      breakdown.overall_ms > 0.0 ? breakdown.overall_ms : breakdown.total_ms);
  (void)recv_tp;

  ModelStatsKey key{std::string(resolved_model_name), request->model_version()};
  std::scoped_lock lock(model_stats_mutex_);
  auto& stats = model_stats_[key];
  stats.last_inference_ms = now_ms;
  stats.inference_count += batch_size;
  stats.execution_count += 1;
  stats.inference_stats.success.count += 1;
  stats.inference_stats.success.ns += total_ns;
  stats.inference_stats.queue.count += 1;
  stats.inference_stats.queue.ns += duration_ms_to_ns(breakdown.queue_ms);
  stats.inference_stats.compute_input.count += 1;
  stats.inference_stats.compute_input.ns +=
      duration_ms_to_ns(breakdown.preprocess_ms);
  stats.inference_stats.compute_infer.count += 1;
  stats.inference_stats.compute_infer.ns +=
      duration_ms_to_ns(breakdown.inference_ms);
  stats.inference_stats.compute_output.count += 1;
  stats.inference_stats.compute_output.ns +=
      duration_ms_to_ns(breakdown.postprocess_ms);
}

void
InferenceServiceImpl::record_failure(
    const ModelInferRequest* request, MonotonicClock::time_point recv_tp,
    std::string_view resolved_model_name)
{
  if (request == nullptr) {
    return;
  }
  const auto now_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  const uint64_t total_ns = elapsed_since(recv_tp);

  ModelStatsKey key{std::string(resolved_model_name), request->model_version()};
  std::scoped_lock lock(model_stats_mutex_);
  auto& stats = model_stats_[key];
  stats.last_inference_ms = now_ms;
  stats.inference_stats.fail.count += 1;
  stats.inference_stats.fail.ns += total_ns;
}

#include "inference_service_async_server.inl"
}  // namespace starpu_server
