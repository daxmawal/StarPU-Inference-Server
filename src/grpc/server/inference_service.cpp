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
struct ModelStatisticsTestHooks {
  bool force_null_stat_target = false;
};

auto
handle_model_infer_async_test_hooks()
    -> InferenceServiceImpl::HandleModelInferAsyncTestHooks&
{
  static InferenceServiceImpl::HandleModelInferAsyncTestHooks hooks{};
  return hooks;
}

auto
handle_async_infer_completion_test_hooks()
    -> InferenceServiceImpl::HandleAsyncInferCompletionTestHooks&
{
  static InferenceServiceImpl::HandleAsyncInferCompletionTestHooks hooks{};
  return hooks;
}

auto
model_statistics_test_hooks() -> ModelStatisticsTestHooks&
{
  static ModelStatisticsTestHooks hooks{};
  return hooks;
}
#endif  // SONAR_IGNORE_END
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

#include "inference_service_io_validation.inl"

auto
build_job_failure_result(InferenceJob& job)
    -> std::pair<Status, std::optional<InferenceServiceImpl::AsyncFailureInfo>>
{
  std::optional<InferenceServiceImpl::AsyncFailureInfo> failure_info;
  Status status = Status::OK;
  if (auto job_failure = job.take_failure_info()) {
    const std::string reason = job_failure->reason;
    const std::string detail_message = job_failure->message;
    std::string message;
    if (!reason.empty() && !detail_message.empty()) {
      message =
          std::format("Inference failed ({}): {}", reason, detail_message);
    } else if (!reason.empty()) {
      message = std::format("Inference failed ({})", reason);
    } else if (!detail_message.empty()) {
      message = std::format("Inference failed: {}", detail_message);
    } else {
      message = "Inference failed";
    }
    status = {grpc::StatusCode::INTERNAL, message};
    InferenceServiceImpl::AsyncFailureInfo info{};
    info.stage = std::move(job_failure->stage);
    info.reason = std::move(job_failure->reason);
    info.metrics_reported = job_failure->metrics_reported;
    failure_info = std::move(info);
  } else {
    InferenceServiceImpl::AsyncFailureInfo info{};
    info.stage = "execution";
    info.reason = "empty_output";
    failure_info = std::move(info);
    status = {grpc::StatusCode::INTERNAL, "Inference failed"};
  }
  return {status, std::move(failure_info)};
}

template <typename Callback>
void
handle_async_job_completion(
    InferenceJob& job, Callback&& callback, std::vector<torch::Tensor> outs,
    double latency_ms)
{
  const auto info = job.timing_info_snapshot();
  const auto base = detail::compute_latency_breakdown(info, latency_ms);
  InferenceServiceImpl::LatencyBreakdown timing{};
  timing.queue_ms = base.queue_ms;
  timing.batch_ms = base.batch_ms;
  timing.submit_ms = base.submit_ms;
  timing.scheduling_ms = base.scheduling_ms;
  timing.codelet_ms = base.codelet_ms;
  timing.inference_ms = base.inference_ms;
  timing.callback_ms = base.callback_ms;
  timing.total_ms = base.total_ms;
  detail::TimingInfo copied_info = info;

  if (outs.empty()) {
    auto [status, failure_info] = build_job_failure_result(job);
    std::forward<Callback>(callback)(
        status, {}, timing, copied_info, std::move(failure_info));
    return;
  }

  std::forward<Callback>(callback)(
      Status::OK, std::move(outs), timing, copied_info, std::nullopt);
}
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

void
InferenceServiceImpl::validate_schema_or_throw() const
{
  for (std::size_t i = 0; i < expected_input_types_.size(); ++i) {
    const at::ScalarType dtype = expected_input_types_[i];
    if (scalar_type_to_model_dtype(dtype) ==
        inference::DataType::TYPE_INVALID) {
      throw std::invalid_argument(std::format(
          "Invalid schema: unsupported input datatype at index {}", i));
    }
    (void)scalar_type_to_datatype(dtype);
  }

  if (reference_outputs_ == nullptr) {
    return;
  }

  for (std::size_t i = 0; i < reference_outputs_->size(); ++i) {
    const at::ScalarType dtype = (*reference_outputs_)[i].scalar_type();
    if (scalar_type_to_model_dtype(dtype) ==
        inference::DataType::TYPE_INVALID) {
      throw std::invalid_argument(std::format(
          "Invalid schema: unsupported output datatype at index {}", i));
    }
    (void)scalar_type_to_datatype(dtype);
  }
}

auto
InferenceServiceImpl::ServerLive(
    ServerContext* /*context*/, const ServerLiveRequest* /*request*/,
    ServerLiveResponse* reply) -> Status
{
  reply->set_live(true);
  return Status::OK;
}

auto
InferenceServiceImpl::ServerReady(
    ServerContext* /*context*/, const ServerReadyRequest* /*request*/,
    ServerReadyResponse* reply) -> Status
{
  const bool ready = queue_ != nullptr && !queue_->is_shutdown();
  reply->set_ready(ready);
  return Status::OK;
}

auto
InferenceServiceImpl::ModelReady(
    ServerContext* /*context*/, const ModelReadyRequest* request,
    ModelReadyResponse* reply) -> Status
{
  bool ready = queue_ != nullptr && !queue_->is_shutdown();
  if (ready && !default_model_name_.empty()) {
    const auto& requested_name = request->name();
    if (!requested_name.empty() && requested_name != default_model_name_) {
      ready = false;
    }
  }
  reply->set_ready(ready);
  return Status::OK;
}

auto
InferenceServiceImpl::ServerMetadata(
    ServerContext* /*context*/,
    const inference::ServerMetadataRequest* /*request*/,
    inference::ServerMetadataResponse* reply) -> Status
{
  std::string name = server_name_;
  if (name.empty()) {
    name = default_model_name_;
  }
  if (name.empty()) {
    name = "starpu_server";
  }
  reply->set_name(std::move(name));
  if (!server_version_.empty()) {
    reply->set_version(server_version_);
  }
  return Status::OK;
}

auto
InferenceServiceImpl::ModelMetadata(
    ServerContext* /*context*/, const inference::ModelMetadataRequest* request,
    inference::ModelMetadataResponse* reply) -> Status
{
  const auto resolved_model_name = resolve_model_name(request->name());
  if (!resolved_model_name.empty()) {
    reply->set_name(resolved_model_name);
  }
  if (!request->version().empty()) {
    reply->add_versions(request->version());
  }

  for (std::size_t i = 0; i < expected_input_types_.size(); ++i) {
    auto* input = reply->add_inputs();
    input->set_name(resolve_tensor_name(i, expected_input_names_, "input"));
    try {
      input->set_datatype(scalar_type_to_datatype(expected_input_types_[i]));
    }
    catch (const std::invalid_argument& e) {
      return {grpc::StatusCode::INTERNAL, e.what()};
    }
    if (i < expected_input_dims_.size()) {
      for (const auto dim : expected_input_dims_[i]) {
        input->add_shape(dim);
      }
    }
  }

  if (reference_outputs_ != nullptr) {
    for (std::size_t i = 0; i < reference_outputs_->size(); ++i) {
      const auto& output = (*reference_outputs_)[i];
      auto* output_meta = reply->add_outputs();
      output_meta->set_name(
          resolve_tensor_name(i, expected_output_names_, "output"));
      try {
        output_meta->set_datatype(
            scalar_type_to_datatype(output.scalar_type()));
      }
      catch (const std::invalid_argument& e) {
        return {grpc::StatusCode::INTERNAL, e.what()};
      }
      for (const auto dim : output.sizes()) {
        output_meta->add_shape(dim);
      }
    }
  }

  return Status::OK;
}

auto
InferenceServiceImpl::ModelConfig(
    ServerContext* /*context*/, const inference::ModelConfigRequest* request,
    inference::ModelConfigResponse* reply) -> Status
{
  auto* config = reply->mutable_config();
  const auto resolved_model_name = resolve_model_name(request->name());
  if (!resolved_model_name.empty()) {
    config->set_name(resolved_model_name);
  }
  config->set_max_batch_size(max_batch_size_);

  for (std::size_t i = 0; i < expected_input_types_.size(); ++i) {
    auto* input = config->add_input();
    input->set_name(resolve_tensor_name(i, expected_input_names_, "input"));
    const auto dtype = scalar_type_to_model_dtype(expected_input_types_[i]);
    if (dtype == inference::DataType::TYPE_INVALID) {
      return {grpc::StatusCode::INTERNAL, "Unsupported input datatype"};
    }
    input->set_data_type(dtype);
    if (i < expected_input_dims_.size()) {
      for (const auto dim : expected_input_dims_[i]) {
        input->add_dims(dim);
      }
    }
  }

  if (reference_outputs_ != nullptr) {
    for (std::size_t i = 0; i < reference_outputs_->size(); ++i) {
      const auto& output = (*reference_outputs_)[i];
      auto* output_config = config->add_output();
      output_config->set_name(
          resolve_tensor_name(i, expected_output_names_, "output"));
      const auto dtype = scalar_type_to_model_dtype(output.scalar_type());
      if (dtype == inference::DataType::TYPE_INVALID) {
        return {grpc::StatusCode::INTERNAL, "Unsupported output datatype"};
      }
      output_config->set_data_type(dtype);
      for (const auto dim : output.sizes()) {
        output_config->add_dims(dim);
      }
    }
  }

  return Status::OK;
}

auto
InferenceServiceImpl::ModelStatistics(
    ServerContext* /*context*/,
    const inference::ModelStatisticsRequest* request,
    inference::ModelStatisticsResponse* reply) -> Status
{
  if (request == nullptr || reply == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "Invalid request"};
  }
  const std::string requested_name = resolve_model_name(request->name());
  const std::string requested_version = request->version();
  const auto fill_stat = [](inference::StatisticDuration* target,
                            const StatisticDurationState& state) {
    if (target == nullptr) {
      return;
    }
    target->set_count(state.count);
    target->set_ns(state.ns);
  };

  std::vector<std::pair<ModelStatsKey, ModelStatsState>> snapshot;
  {
    std::scoped_lock lock(model_stats_mutex_);
    snapshot.reserve(model_stats_.size());
    for (const auto& [key, state] : model_stats_) {
      snapshot.emplace_back(key, state);
    }
  }

  for (const auto& [key, state] : snapshot) {
    if (!requested_name.empty() && key.name != requested_name) {
      continue;
    }
    if (!requested_version.empty() && key.version != requested_version) {
      continue;
    }
    auto* stats = reply->add_model_stats();
    if (!key.name.empty()) {
      stats->set_name(key.name);
    }
    if (!key.version.empty()) {
      stats->set_version(key.version);
    }
    stats->set_last_inference(state.last_inference_ms);
    stats->set_inference_count(state.inference_count);
    stats->set_execution_count(state.execution_count);

    auto* infer_stats = stats->mutable_inference_stats();
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (model_statistics_test_hooks().force_null_stat_target) {
      fill_stat(nullptr, state.inference_stats.success);
    }
#endif  // SONAR_IGNORE_END
    fill_stat(infer_stats->mutable_success(), state.inference_stats.success);
    fill_stat(infer_stats->mutable_fail(), state.inference_stats.fail);
    fill_stat(infer_stats->mutable_queue(), state.inference_stats.queue);
    fill_stat(
        infer_stats->mutable_compute_input(),
        state.inference_stats.compute_input);
    fill_stat(
        infer_stats->mutable_compute_infer(),
        state.inference_stats.compute_infer);
    fill_stat(
        infer_stats->mutable_compute_output(),
        state.inference_stats.compute_output);
  }

  return Status::OK;
}

auto
InferenceServiceImpl::ModelStreamInfer(
    ServerContext* /*context*/,
    grpc::ServerReaderWriter<
        inference::ModelStreamInferResponse,
        inference::ModelInferRequest>* /*stream*/) -> Status
{
  return unimplemented_rpc_status("ModelStreamInfer");
}

auto
InferenceServiceImpl::RepositoryIndex(
    ServerContext* /*context*/,
    const inference::RepositoryIndexRequest* /*request*/,
    inference::RepositoryIndexResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("RepositoryIndex");
}

auto
InferenceServiceImpl::RepositoryModelLoad(
    ServerContext* /*context*/,
    const inference::RepositoryModelLoadRequest* /*request*/,
    inference::RepositoryModelLoadResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("RepositoryModelLoad");
}

auto
InferenceServiceImpl::RepositoryModelUnload(
    ServerContext* /*context*/,
    const inference::RepositoryModelUnloadRequest* /*request*/,
    inference::RepositoryModelUnloadResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("RepositoryModelUnload");
}

auto
InferenceServiceImpl::SystemSharedMemoryStatus(
    ServerContext* /*context*/,
    const inference::SystemSharedMemoryStatusRequest* /*request*/,
    inference::SystemSharedMemoryStatusResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("SystemSharedMemoryStatus");
}

auto
InferenceServiceImpl::SystemSharedMemoryRegister(
    ServerContext* /*context*/,
    const inference::SystemSharedMemoryRegisterRequest* /*request*/,
    inference::SystemSharedMemoryRegisterResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("SystemSharedMemoryRegister");
}

auto
InferenceServiceImpl::SystemSharedMemoryUnregister(
    ServerContext* /*context*/,
    const inference::SystemSharedMemoryUnregisterRequest* /*request*/,
    inference::SystemSharedMemoryUnregisterResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("SystemSharedMemoryUnregister");
}

auto
InferenceServiceImpl::CudaSharedMemoryStatus(
    ServerContext* /*context*/,
    const inference::CudaSharedMemoryStatusRequest* /*request*/,
    inference::CudaSharedMemoryStatusResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("CudaSharedMemoryStatus");
}

auto
InferenceServiceImpl::CudaSharedMemoryRegister(
    ServerContext* /*context*/,
    const inference::CudaSharedMemoryRegisterRequest* /*request*/,
    inference::CudaSharedMemoryRegisterResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("CudaSharedMemoryRegister");
}

auto
InferenceServiceImpl::CudaSharedMemoryUnregister(
    ServerContext* /*context*/,
    const inference::CudaSharedMemoryUnregisterRequest* /*request*/,
    inference::CudaSharedMemoryUnregisterResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("CudaSharedMemoryUnregister");
}

auto
InferenceServiceImpl::TraceSetting(
    ServerContext* /*context*/,
    const inference::TraceSettingRequest* /*request*/,
    inference::TraceSettingResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("TraceSetting");
}

auto
InferenceServiceImpl::LogSettings(
    ServerContext* /*context*/,
    const inference::LogSettingsRequest* /*request*/,
    inference::LogSettingsResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("LogSettings");
}

auto
InferenceServiceImpl::validate_and_convert_inputs(
    const ModelInferRequest* request, std::vector<torch::Tensor>& inputs,
    std::vector<std::shared_ptr<const void>>* input_lifetimes) -> Status
{
  if (request == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "ModelInfer request is null"};
  }
  if (auto status =
          validate_input_counts(request, expected_input_types_.size());
      !status.ok()) {
    return status;
  }

  const auto name_state = collect_input_name_state(request);
  const bool has_expected_names = !expected_input_names_.empty();
  if (auto status = validate_input_names(name_state, has_expected_names);
      !status.ok()) {
    return status;
  }
  const bool use_name_mapping = has_expected_names;

  std::unordered_map<std::string_view, std::size_t> expected_index_by_name;
  if (use_name_mapping) {
    if (auto status = build_expected_index_by_name(
            expected_input_names_, expected_index_by_name);
        !status.ok()) {
      return status;
    }
  }

  const std::size_t expected_count = expected_input_types_.size();
  std::vector<torch::Tensor> ordered_inputs(expected_count);
  std::vector<std::shared_ptr<const void>> ordered_lifetimes;
  if (input_lifetimes != nullptr) {
    ordered_lifetimes.resize(expected_count);
  }
  std::vector<bool> filled(expected_count, false);
  const ProcessInputContext process_context{
      use_name_mapping ? &expected_index_by_name : nullptr,
      &expected_input_types_,
      &expected_input_dims_,
      max_batch_size_,
      &ordered_inputs,
      input_lifetimes != nullptr ? &ordered_lifetimes : nullptr,
      &filled};

  for (int i = 0; i < request->inputs_size(); ++i) {
    Status status = process_input(request, i, process_context);
    if (!status.ok()) {
      return status;
    }
  }

  if (use_name_mapping) {
    Status status = check_missing_named_inputs(filled, expected_input_names_);
    if (!status.ok()) {
      return status;
    }
  }

  inputs = std::move(ordered_inputs);
  if (input_lifetimes != nullptr) {
    *input_lifetimes = std::move(ordered_lifetimes);
  }
  return Status::OK;
}

auto
InferenceServiceImpl::build_latency_breakdown(
    const detail::TimingInfo& info, double latency_ms) -> LatencyBreakdown
{
  const auto base = detail::compute_latency_breakdown(info, latency_ms);
  LatencyBreakdown breakdown{};
  breakdown.queue_ms = base.queue_ms;
  breakdown.batch_ms = base.batch_ms;
  breakdown.submit_ms = base.submit_ms;
  breakdown.scheduling_ms = base.scheduling_ms;
  breakdown.codelet_ms = base.codelet_ms;
  breakdown.inference_ms = base.inference_ms;
  breakdown.callback_ms = base.callback_ms;
  breakdown.total_ms = base.total_ms;
  return breakdown;
}

auto
InferenceServiceImpl::resolve_model_name(std::string model_name) const
    -> std::string
{
  if (default_model_name_.empty()) {
    return model_name;
  }
  if (!model_name.empty() && model_name != default_model_name_) {
    log_warning(std::format(
        "Client requested model '{}' but server is configured for '{}'; "
        "defaulting to server model",
        model_name, default_model_name_));
  }
  return default_model_name_;
}

auto
InferenceServiceImpl::next_request_id() -> int
{
  constexpr int kMaxRequestId = std::numeric_limits<int>::max();
  int current = next_request_id_.load(std::memory_order_relaxed);
  while (true) {
    int issued = 0;
    int next = 1;
    if (current >= 0 && current < kMaxRequestId) {
      issued = current;
      next = current + 1;
    }
    if (next_request_id_.compare_exchange_weak(
            current, next, std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
      return issued;
    }
  }
}

auto
InferenceServiceImpl::
    submit_job_async(  // NOLINT(readability-function-cognitive-complexity)
        const std::vector<torch::Tensor>& inputs, AsyncJobCallback on_complete,
        std::vector<std::shared_ptr<const void>> input_lifetimes,
        std::shared_ptr<std::atomic<bool>> cancel_flag,
        MonotonicClock::time_point receive_time, std::string model_name,
        std::optional<AsyncFailureInfo>* submit_failure_info) -> Status
{
  auto resolved_model_name = resolve_model_name(std::move(model_name));
  if (submit_failure_info != nullptr) {
    *submit_failure_info = std::nullopt;
  }
  const auto set_submit_failure_info = [&](std::string_view reason) {
    if (submit_failure_info == nullptr) {
      return;
    }
    AsyncFailureInfo info{};
    info.stage = "enqueue";
    info.reason = std::string(reason);
    *submit_failure_info = std::move(info);
  };

  if (queue_ == nullptr) {
    set_submit_failure_info("queue_unavailable");
    return {grpc::StatusCode::UNAVAILABLE, "Inference queue unavailable"};
  }
  if (reference_outputs_ == nullptr) {
    set_submit_failure_info("reference_outputs_unavailable");
    return {
        grpc::StatusCode::FAILED_PRECONDITION,
        "Reference outputs are unavailable"};
  }

  try {
    auto job = client_utils::create_job(
        inputs, *reference_outputs_, next_request_id(),
        std::move(input_lifetimes), receive_time,
        std::string(resolved_model_name));
    job->set_cancelled_flag(std::move(cancel_flag));

    NvtxRange submit_scope("grpc_submit_starpu");

    job->set_on_complete([job, callback = std::move(on_complete)](
                             std::vector<torch::Tensor> outs,
                             double latency_ms) mutable {
      try {
        handle_async_job_completion(
            *job, callback, std::move(outs), latency_ms);
      }
      catch (const std::exception& e) {
        log_error(std::format(
            "Unhandled exception while dispatching async completion: {}",
            e.what()));
        InferenceServiceImpl::AsyncFailureInfo failure_info{
            .stage = "completion", .reason = "exception"};
        try {
          callback(
              {grpc::StatusCode::INTERNAL,
               "Internal server error during async completion"},
              {}, InferenceServiceImpl::LatencyBreakdown{},
              detail::TimingInfo{}, std::move(failure_info));
        }
        catch (const std::exception& callback_exception) {
          log_error(std::format(
              "Unhandled exception while reporting async completion failure: "
              "{}",
              callback_exception.what()));
        }
        catch (...) {
          log_error(
              "Unhandled non-std exception while reporting async completion "
              "failure");
        }
      }
      catch (...) {
        log_error(
            "Unhandled non-std exception while dispatching async completion");
        InferenceServiceImpl::AsyncFailureInfo failure_info{
            .stage = "completion", .reason = "unknown_exception"};
        try {
          callback(
              {grpc::StatusCode::INTERNAL,
               "Internal server error during async completion"},
              {}, InferenceServiceImpl::LatencyBreakdown{},
              detail::TimingInfo{}, std::move(failure_info));
        }
        catch (const std::exception& callback_exception) {
          log_error(std::format(
              "Unhandled exception while reporting async completion failure: "
              "{}",
              callback_exception.what()));
        }
        catch (...) {
          log_error(
              "Unhandled non-std exception while reporting async completion "
              "failure");
        }
      }
    });

    const auto enqueued_now = MonotonicClock::now();
    job->update_timing_info([enqueued_now](detail::TimingInfo& timing) {
      timing.enqueued_time = enqueued_now;
      timing.last_enqueued_time = enqueued_now;
    });

    bool pushed = false;
    bool queue_full = false;
    {
      NvtxRange queue_scope("grpc_submit_starpu_queue");
      pushed = queue_->push(job, &queue_full);
    }
    if (!pushed) {
      if (queue_full) {
        set_submit_failure_info("queue_full");
        increment_rejected_requests();
        congestion::record_rejection(1);
        BatchingTraceLogger::instance().log_request_rejected(queue_->size());
        return {
            grpc::StatusCode::RESOURCE_EXHAUSTED, "Inference queue is full"};
      }
      set_submit_failure_info("queue_unavailable");
      return {grpc::StatusCode::UNAVAILABLE, "Inference queue unavailable"};
    }
    if (auto& tracer = BatchingTraceLogger::instance(); tracer.enabled()) {
      const auto timing = job->timing_info_snapshot();
      tracer.log_request_enqueued(
          job->get_request_id(), job->model_name(), /*is_warmup=*/false,
          timing.last_enqueued_time);
    }
    return Status::OK;
  }
  catch (const std::exception& e) {
    log_error(std::format(
        "Unhandled exception while submitting inference job: {}", e.what()));
    set_submit_failure_info("exception");
    return {grpc::StatusCode::INTERNAL, "Internal server error"};
  }
  catch (...) {
    log_error("Unhandled non-std exception while submitting inference job");
    set_submit_failure_info("unknown_exception");
    return {grpc::StatusCode::INTERNAL, "Internal server error"};
  }
}

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
  auto& test_hooks = handle_model_infer_async_test_hooks();
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

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto
InferenceServiceImpl::submit_job_and_wait(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs, LatencyBreakdown& breakdown,
    detail::TimingInfo& timing_info,
    std::vector<std::shared_ptr<const void>> input_lifetimes) -> Status
{
  struct JobResult {
    Status status = Status::OK;
    std::vector<torch::Tensor> outputs;
    LatencyBreakdown breakdown;
    detail::TimingInfo timing_info;
  };

  auto result_promise = std::make_shared<std::promise<JobResult>>();
  auto result_future = result_promise->get_future();

  const auto receive_time = MonotonicClock::now();
  if (Status submit_status = submit_job_async(
          inputs,
          [result_promise](
              Status status, std::vector<torch::Tensor> outs,
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
  return Status::OK;
}

void
InferenceServiceImpl::TestAccessor::SetHandleModelInferAsyncTestHooks(
    HandleModelInferAsyncTestHooks hooks)
{
  handle_model_infer_async_test_hooks() = std::move(hooks);
}

void
InferenceServiceImpl::TestAccessor::ClearHandleModelInferAsyncTestHooks()
{
  handle_model_infer_async_test_hooks() = HandleModelInferAsyncTestHooks{};
}

void
InferenceServiceImpl::TestAccessor::SetHandleAsyncInferCompletionTestHooks(
    HandleAsyncInferCompletionTestHooks hooks)
{
  handle_async_infer_completion_test_hooks() = std::move(hooks);
}

void
InferenceServiceImpl::TestAccessor::ClearHandleAsyncInferCompletionTestHooks()
{
  handle_async_infer_completion_test_hooks() =
      HandleAsyncInferCompletionTestHooks{};
}

auto
InferenceServiceImpl::TestAccessor::NormalizeNamesForTest(
    std::vector<std::string> names, std::size_t expected_size,
    std::string_view fallback_prefix,
    std::string_view kind) -> std::vector<std::string>
{
  return normalize_names(
      std::move(names), expected_size,
      NormalizeNamesOptions{fallback_prefix, kind});
}

void
InferenceServiceImpl::TestAccessor::SetExpectedInputNamesForTest(
    InferenceServiceImpl* service, std::vector<std::string> names)
{
  service->expected_input_names_ = std::move(names);
}

auto
InferenceServiceImpl::TestAccessor::CheckMissingInputsForTest(
    const std::vector<bool>& filled,
    std::span<const std::string> expected_names) -> grpc::Status
{
  return check_missing_named_inputs(filled, expected_names);
}

void
InferenceServiceImpl::TestAccessor::ArmRpcDoneTagWithNullContextForTest()
{
  auto tag = RpcDoneTag::Create([] {}, std::make_shared<int>(0));
  tag->Arm(nullptr);
}

auto
InferenceServiceImpl::TestAccessor::RpcDoneTagProceedForTest(
    bool is_ok, bool with_on_done) -> bool
{
  bool called = false;
  RpcDoneTag::OnDone on_done =
      with_on_done ? [&called]() { called = true; } : RpcDoneTag::OnDone{};
  auto tag = RpcDoneTag::Create(std::move(on_done), std::make_shared<int>(0));
  tag->Proceed(is_ok);
  return called;
}

void
InferenceServiceImpl::TestAccessor::SetGrpcHealthStatusForTest(
    grpc::Server* server, bool serving)
{
  set_grpc_health_status(server, serving);
}

void
InferenceServiceImpl::TestAccessor::RecordSuccessForTest(
    InferenceServiceImpl* service, const ModelInferRequest* request,
    const LatencyBreakdown& breakdown, MonotonicClock::time_point recv_tp,
    std::string_view resolved_model_name)
{
  service->record_success(request, breakdown, recv_tp, resolved_model_name);
}

void
InferenceServiceImpl::TestAccessor::RecordFailureForTest(
    InferenceServiceImpl* service, const ModelInferRequest* request,
    MonotonicClock::time_point recv_tp, std::string_view resolved_model_name)
{
  service->record_failure(request, recv_tp, resolved_model_name);
}

auto
InferenceServiceImpl::TestAccessor::ScalarTypeToModelDtypeForTest(
    at::ScalarType type) -> inference::DataType
{
  return scalar_type_to_model_dtype(type);
}

auto
InferenceServiceImpl::TestAccessor::ResolveTensorNameForTest(
    std::size_t index, std::span<const std::string> names,
    std::string_view fallback_prefix) -> std::string
{
  return resolve_tensor_name(index, names, fallback_prefix);
}

auto
InferenceServiceImpl::TestAccessor::RequestBatchSizeForTest(
    const ModelInferRequest* request, int max_batch_size) -> uint64_t
{
  return request_batch_size(request, max_batch_size);
}

auto
InferenceServiceImpl::TestAccessor::DurationMsToNsForTest(double duration_ms)
    -> uint64_t
{
  return duration_ms_to_ns(duration_ms);
}

auto
InferenceServiceImpl::TestAccessor::ElapsedSinceForTest(
    MonotonicClock::time_point start) -> uint64_t
{
  return elapsed_since(start);
}

void
InferenceServiceImpl::TestAccessor::SetModelStatisticsForceNullTargetForTest(
    bool enable)
{
  model_statistics_test_hooks().force_null_stat_target = enable;
}

auto
InferenceServiceImpl::TestAccessor::IsContextCancelledForTest(
    grpc::ServerContext* context) -> bool
{
  return is_context_cancelled(context);
}

auto
InferenceServiceImpl::TestAccessor::FillOutputTensorForTest(
    inference::ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs,
    const std::vector<std::size_t>& output_indices,
    const std::vector<std::string>& output_names) -> grpc::Status
{
  return fill_output_tensor(reply, outputs, output_indices, output_names);
}

auto
InferenceServiceImpl::TestAccessor::BuildLatencyBreakdownForTest(
    const detail::TimingInfo& info, double latency_ms) -> LatencyBreakdown
{
  return build_latency_breakdown(info, latency_ms);
}

auto
InferenceServiceImpl::TestAccessor::HandleAsyncInferCompletionForTest(
    bool cancelled) -> bool
{
  inference::ModelInferRequest request;
  inference::ModelInferResponse reply;
  std::vector<torch::Tensor> outputs = {
      torch::zeros({1}, torch::TensorOptions().dtype(at::kFloat))};
  auto cancel_flag = std::make_shared<std::atomic<bool>>(cancelled);
  auto terminal_flag = std::make_shared<std::atomic<bool>>(false);
  bool called = false;
  auto callback_handle = std::make_shared<CallbackHandle>(
      [&called](Status /*unused*/) { called = true; });
  LatencyBreakdown breakdown{};
  detail::TimingInfo timing_info{};
  AsyncInferCompletionContext context{
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
  handle_async_infer_completion(
      context, Status::OK, outputs, breakdown, timing_info);
  return called;
}
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

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
  auto& async_hooks = handle_async_infer_completion_test_hooks();
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
  auto& async_hooks = handle_async_infer_completion_test_hooks();
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
  auto& test_hooks = handle_model_infer_async_test_hooks();
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
  auto& test_hooks = handle_model_infer_async_test_hooks();
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
    if (!try_mark_terminal(terminal_flag)) {
      return;
    }
    const Status cancelled_status{
        grpc::StatusCode::CANCELLED, "Request cancelled"};
    const bool invoked = callback_handle->Invoke(cancelled_status);
    if (invoked) {
      if (service != nullptr) {
        service->record_failure(request, recv_tp, model_name);
      }
      record_terminal_metrics(
          model_name, cancelled_status, "cancel", "client_cancelled");
    }
  };

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  auto& test_hooks = handle_model_infer_async_test_hooks();
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
  const auto& cancel_flag = terminal_state.cancel_flag;
  const auto& terminal_flag = terminal_state.terminal_flag;
  if (cancel_flag != nullptr && cancel_flag->load(std::memory_order_acquire)) {
    return true;
  }
  if (!try_mark_terminal(terminal_flag)) {
    return true;
  }
  if (service != nullptr) {
    service->record_failure(request, recv_tp, resolved_model_name);
  }
  record_terminal_metrics(resolved_model_name, status, "preprocess");
  if (callback_handle != nullptr) {
    callback_handle->Invoke(status);
  }
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
  const auto& cancel_flag = terminal_state.cancel_flag;
  const auto& terminal_flag = terminal_state.terminal_flag;
  if (cancel_flag != nullptr && cancel_flag->load(std::memory_order_acquire)) {
    return true;
  }
  if (!try_mark_terminal(terminal_flag)) {
    return true;
  }
  if (service != nullptr) {
    service->record_failure(request, recv_tp, resolved_model_name);
  }
  record_terminal_metrics(
      resolved_model_name, status, "enqueue", {}, failure_info);
  if (callback_handle != nullptr) {
    callback_handle->Invoke(status);
  }
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
  const auto& cancel_flag = terminal_state.cancel_flag;
  const auto& terminal_flag = terminal_state.terminal_flag;
  if (cancel_flag != nullptr && cancel_flag->load(std::memory_order_acquire)) {
    return;
  }
  if (!try_mark_terminal(terminal_flag)) {
    return;
  }

  const Status status{grpc::StatusCode::INTERNAL, "Internal server error"};
  const bool invoked =
      callback_handle != nullptr && callback_handle->Invoke(status);
  if (!invoked) {
    return;
  }

  if (service != nullptr) {
    service->record_failure(request, recv_tp, resolved_model_name);
  }
  record_terminal_metrics(
      resolved_model_name, status, details.stage, details.reason);
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

void
InferenceServiceImpl::HandleModelInferAsync(
    ServerContext* context, const ModelInferRequest* request,
    ModelInferResponse* reply, std::function<void(Status)> on_done,
    std::shared_ptr<void> call_guard)
{
  auto callback_handle = std::make_shared<CallbackHandle>(std::move(on_done));
  auto cancel_flag = std::make_shared<std::atomic<bool>>(false);
  auto terminal_flag = std::make_shared<std::atomic<bool>>(false);
  const AsyncTerminalState terminal_state{
      .cancel_flag = cancel_flag, .terminal_flag = terminal_flag};
  notify_cancel_flag_created(cancel_flag);
  if (auto guard_status = validate_model_infer_io(request, reply);
      !guard_status.ok()) {
    callback_handle->Invoke(guard_status);
    return;
  }

  const auto resolved_model_name = resolve_model_name(request->model_name());
  auto recv_tp = MonotonicClock::now();

  try {
    NvtxRange request_scope("grpc_handle_infer_request");

    auto metrics = get_metrics();
    if (metrics && metrics->counters().requests_total != nullptr) {
      metrics->counters().requests_total->Increment();
    }
    increment_requests_received(resolved_model_name);
    congestion::record_arrival(1);
    int64_t recv_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
    if (setup_async_cancellation(
            context, call_guard,
            AsyncCancellationContext{
                .cancel_flag = cancel_flag,
                .terminal_flag = terminal_flag,
                .callback_handle = callback_handle,
                .resolved_model_name = resolved_model_name,
                .service = this,
                .request = request,
                .recv_tp = recv_tp})) {
      return;
    }

    std::vector<torch::Tensor> inputs;
    std::vector<std::shared_ptr<const void>> input_lifetimes;
    std::optional<AsyncFailureInfo> submit_failure_info;
    Status status =
        validate_and_convert_inputs(request, inputs, &input_lifetimes);
    if (handle_input_validation_failure(
            status, terminal_state, callback_handle, this, resolved_model_name,
            request, recv_tp)) {
      return;
    }

    if (cancel_flag->load(std::memory_order_acquire)) {
      return;
    }

    const auto* output_names = &expected_output_names_;
    status = submit_job_async(
        inputs,
        [request, reply, recv_tp, recv_ms, metrics, callback_handle,
         resolved_model_name, cancel_flag, terminal_flag, terminal_state,
         output_names, service = this](
            Status const& job_status, const std::vector<torch::Tensor>& outs,
            LatencyBreakdown breakdown, detail::TimingInfo timing_info,
            std::optional<AsyncFailureInfo> failure_info) mutable {
          try {
            handle_async_infer_completion(
                AsyncInferCompletionContext{
                    request, reply, callback_handle, metrics, recv_tp, recv_ms,
                    resolved_model_name, service, output_names, cancel_flag,
                    terminal_flag, std::move(failure_info)},
                job_status, outs, breakdown, timing_info);
          }
          catch (const std::exception& e) {
            handle_async_internal_error(
                terminal_state, callback_handle, service, resolved_model_name,
                request, recv_tp,
                AsyncInternalErrorDetails{
                    .stage = "postprocess",
                    .reason = "exception",
                    .log_context = std::format(
                        "Unhandled exception in async inference completion: {}",
                        e.what())});
          }
          catch (...) {
            handle_async_internal_error(
                terminal_state, callback_handle, service, resolved_model_name,
                request, recv_tp,
                AsyncInternalErrorDetails{
                    .stage = "postprocess",
                    .reason = "unknown_exception",
                    .log_context =
                        "Unhandled non-std exception in async inference "
                        "completion"});
          }
        },
        std::move(input_lifetimes), cancel_flag, recv_tp, resolved_model_name,
        &submit_failure_info);

    notify_submit_job_async_done(cancel_flag, status);
    if (handle_submit_failure(
            status, terminal_state, callback_handle, this, resolved_model_name,
            request, recv_tp, submit_failure_info)) {
      return;
    }
  }
  catch (const std::exception& e) {
    handle_async_internal_error(
        terminal_state, callback_handle, this, resolved_model_name, request,
        recv_tp,
        AsyncInternalErrorDetails{
            .stage = "internal",
            .reason = "exception",
            .log_context = std::format(
                "Unhandled exception in HandleModelInferAsync: {}", e.what())});
  }
  catch (...) {
    handle_async_internal_error(
        terminal_state, callback_handle, this, resolved_model_name, request,
        recv_tp,
        AsyncInternalErrorDetails{
            .stage = "internal",
            .reason = "unknown_exception",
            .log_context = "Unhandled non-std exception in "
                           "HandleModelInferAsync"});
  }
}

auto
InferenceServiceImpl::populate_response(
    const ModelInferRequest* request, ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs, int64_t recv_ms,
    const LatencyBreakdown& breakdown) -> Status
{
  return populate_response(
      request, reply, outputs, recv_ms, breakdown, PopulateResponseOptions{});
}

auto
InferenceServiceImpl::populate_response(
    const ModelInferRequest* request, ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs, int64_t recv_ms,
    const LatencyBreakdown& breakdown,
    PopulateResponseOptions options) -> Status
{
  if (request == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "ModelInfer request is null"};
  }
  if (reply == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "ModelInfer response is null"};
  }
  if (!options.model_name_override.empty()) {
    reply->set_model_name(std::string(options.model_name_override));
  } else {
    reply->set_model_name(request->model_name());
  }
  reply->set_model_version(request->model_version());
  reply->set_server_receive_ms(recv_ms);
  reply->set_server_queue_ms(breakdown.queue_ms);
  reply->set_server_batch_ms(breakdown.batch_ms);
  reply->set_server_submit_ms(breakdown.submit_ms);
  reply->set_server_scheduling_ms(breakdown.scheduling_ms);
  reply->set_server_codelet_ms(breakdown.codelet_ms);
  reply->set_server_inference_ms(breakdown.inference_ms);
  reply->set_server_callback_ms(breakdown.callback_ms);
  if (options.set_prepost_overall) {
    reply->set_server_preprocess_ms(breakdown.preprocess_ms);
    reply->set_server_postprocess_ms(breakdown.postprocess_ms);
    reply->set_server_overall_ms(breakdown.overall_ms);
  }
  reply->set_server_total_ms(breakdown.total_ms);

  const auto resolved_names =
      resolve_output_names(options.output_names, outputs.size());
  std::vector<std::size_t> output_indices;
  if (request->outputs_size() > 0) {
    std::unordered_map<std::string_view, std::size_t> index_by_name;
    index_by_name.reserve(resolved_names.size());
    for (std::size_t i = 0; i < resolved_names.size(); ++i) {
      const auto [existing_it, inserted] =
          index_by_name.try_emplace(resolved_names[i], i);
      if (!inserted) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format(
                "Configured output name '{}' is duplicated",
                existing_it->first)};
      }
    }

    std::unordered_set<std::string_view> seen;
    output_indices.reserve(request->outputs_size());
    for (const auto& requested : request->outputs()) {
      if (requested.name().empty()) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            "Requested output name must be non-empty"};
      }
      const auto name_iter = index_by_name.find(requested.name());
      if (name_iter == index_by_name.end()) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format(
                "Requested output '{}' is not available", requested.name())};
      }
      if (!seen.insert(name_iter->first).second) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format(
                "Requested output '{}' is duplicated", requested.name())};
      }
      output_indices.push_back(name_iter->second);
    }
  } else {
    output_indices.reserve(outputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      output_indices.push_back(i);
    }
  }

  return fill_output_tensor(reply, outputs, output_indices, resolved_names);
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto
InferenceServiceImpl::ModelInfer(
    ServerContext* context, const ModelInferRequest* request,
    ModelInferResponse* reply) -> Status
{
  std::promise<Status> status_promise;
  auto status_future = status_promise.get_future();
  HandleModelInferAsync(
      context, request, reply, [&status_promise](Status status) {
        status_promise.set_value(std::move(status));
      });
  return status_future.get();
}
#endif  // SONAR_IGNORE_END

#include "inference_service_async_server.inl"
}  // namespace starpu_server
