#pragma once

#include <grpcpp/health_check_service_interface.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
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

#include "core/inference_runner.hpp"
#include "inference_service.hpp"
#include "monitoring/congestion_monitor.hpp"
#include "monitoring/metrics.hpp"
#include "monitoring/runtime_observability.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/client_utils.hpp"
#include "utils/datatype_utils.hpp"
#include "utils/logger.hpp"
#include "utils/nvtx.hpp"

namespace starpu_server {

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
namespace testing::inference_service_test_internal::detail {
auto handle_model_infer_async_test_hooks_ref()
    -> testing::HandleModelInferAsyncTestHooks&;
auto handle_async_infer_completion_test_hooks_ref()
    -> testing::HandleAsyncInferCompletionTestHooks&;
auto submit_job_async_test_hooks_ref() -> testing::SubmitJobAsyncTestHooks&;
auto check_missing_named_inputs_override_ref()
    -> testing::CheckMissingNamedInputsOverrideFn&;
auto model_statistics_force_null_target_flag_ref() -> bool&;
}  // namespace testing::inference_service_test_internal::detail
#endif  // SONAR_IGNORE_END

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

auto unimplemented_rpc_status(std::string_view rpc_name) -> Status;
auto check_missing_named_inputs(
    const std::vector<bool>& filled,
    std::span<const std::string> expected_names) -> Status;
auto scalar_type_to_model_dtype(at::ScalarType type) -> inference::DataType;
auto resolve_tensor_name(
    std::size_t index, std::span<const std::string> names,
    std::string_view fallback_prefix) -> std::string;
auto request_batch_size(const ModelInferRequest* request, int max_batch_size)
    -> uint64_t;
auto validate_model_infer_io(
    const ModelInferRequest* request,
    const ModelInferResponse* reply) -> Status;
auto duration_ms_to_ns(double duration_ms) -> uint64_t;
auto elapsed_since(MonotonicClock::time_point start) -> uint64_t;
auto validate_configured_shape(
    const std::vector<int64_t>& shape, const std::vector<int64_t>& expected,
    bool batching_allowed, int max_batch_size) -> Status;
auto fill_output_tensor(
    ModelInferResponse* reply, const std::vector<torch::Tensor>& outputs,
    std::span<const std::size_t> output_indices,
    std::span<const std::string> output_names) -> Status;
void set_grpc_health_status(const Server* server, bool serving);
auto is_context_cancelled(const ServerContext* context) -> bool;

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

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto unary_call_data_missing_handler_transitions_to_finish_for_test_impl()
    -> bool;
#endif  // SONAR_IGNORE_END

}  // namespace inference_service_detail

}  // namespace starpu_server
