#include <arpa/inet.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <latch>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#define main starpu_server_server_main_for_test
#include "../../../src/grpc/server/server_main.cpp"
#undef main

namespace {

auto
running_under_tsan() -> bool
{
#if defined(__SANITIZE_THREAD__)
  return true;
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
  return true;
#endif
#endif
  return false;
}

struct TempFileGuard {
  std::filesystem::path path;

  ~TempFileGuard()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

struct TempDirectoryGuard {
  std::filesystem::path path;

  ~TempDirectoryGuard()
  {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

class OStreamCapture {
 public:
  explicit OStreamCapture(std::ostream& stream)
      : stream_(stream), old_buffer_(stream.rdbuf(buffer_.rdbuf()))
  {
  }

  ~OStreamCapture() { stream_.rdbuf(old_buffer_); }

  [[nodiscard]] auto str() const -> std::string { return buffer_.str(); }

 private:
  std::ostream& stream_;
  std::ostringstream buffer_;
  std::streambuf* old_buffer_;
};

auto
make_temp_test_directory(std::string_view stem) -> TempDirectoryGuard
{
  static std::atomic<std::uint64_t> temp_directory_counter{0};
  const auto id =
      temp_directory_counter.fetch_add(1, std::memory_order_relaxed);
  const auto path = std::filesystem::temp_directory_path() /
                    (std::string(stem) + "_" + std::to_string(id));
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  EXPECT_FALSE(ec) << "Failed to create temporary directory: " << path;
  return TempDirectoryGuard{path};
}

auto
expected_warning_log(std::string_view message) -> std::string
{
  return std::string("\x1b[1;33m[WARNING] ") + std::string(message) +
         "\x1b[0m\n";
}

auto
expected_info_log(std::string_view message) -> std::string
{
  return std::string("\x1b[1;32m[INFO] ") + std::string(message) + "\x1b[0m\n";
}

struct ScopedTraceLoggerReset {
  ~ScopedTraceLoggerReset()
  {
    starpu_server::BatchingTraceLogger::instance().configure(false, "");
  }
};

struct ScopedResetTraceLoggerForcedThrow {
  explicit ScopedResetTraceLoggerForcedThrow(bool enabled = true) noexcept
  {
    RuntimeCleanupGuard::SetResetTraceLoggerNoexceptForceThrowForTest(enabled);
  }

  ~ScopedResetTraceLoggerForcedThrow()
  {
    RuntimeCleanupGuard::SetResetTraceLoggerNoexceptForceThrowForTest(false);
  }
};

struct ScopedShutdownMetricsForcedThrow {
  explicit ScopedShutdownMetricsForcedThrow(bool enabled = true) noexcept
  {
    RuntimeCleanupGuard::SetShutdownMetricsNoexceptForceThrowForTest(enabled);
  }

  ~ScopedShutdownMetricsForcedThrow()
  {
    RuntimeCleanupGuard::SetShutdownMetricsNoexceptForceThrowForTest(false);
  }
};

struct ScopedSignalPipeForcedPipeFailure {
  explicit ScopedSignalPipeForcedPipeFailure(bool enabled = true) noexcept
  {
    SignalNotificationPipe::SetPipeFailureForTest(enabled);
  }

  ~ScopedSignalPipeForcedPipeFailure()
  {
    SignalNotificationPipe::SetPipeFailureForTest(false);
  }
};

struct ScopedSignalPipeForcedSetNonBlockingFailure {
  explicit ScopedSignalPipeForcedSetNonBlockingFailure(
      bool enabled = true) noexcept
  {
    SignalNotificationPipe::SetSetNonBlockingFailureForTest(enabled);
  }

  ~ScopedSignalPipeForcedSetNonBlockingFailure()
  {
    SignalNotificationPipe::SetSetNonBlockingFailureForTest(false);
  }
};

struct ResolvePythonExecutableOverrideState {
  std::vector<std::filesystem::path> candidates;
  bool is_regular_file_result = false;
  bool force_status_error = false;
  int candidates_calls = 0;
  int is_regular_file_calls = 0;
  std::vector<std::filesystem::path> observed_candidates;
};

auto
resolve_python_executable_override_state()
    -> ResolvePythonExecutableOverrideState*&
{
  static ResolvePythonExecutableOverrideState* state = nullptr;
  return state;
}

auto
resolve_python_candidates_override_stub() -> std::vector<std::filesystem::path>
{
  auto* state = resolve_python_executable_override_state();
  if (state == nullptr) {
    return {};
  }
  ++state->candidates_calls;
  return state->candidates;
}

auto
resolve_python_is_regular_file_override_stub(
    const std::filesystem::path& candidate, std::error_code& status_ec) -> bool
{
  auto* state = resolve_python_executable_override_state();
  if (state == nullptr) {
    status_ec.clear();
    return false;
  }
  ++state->is_regular_file_calls;
  state->observed_candidates.push_back(candidate);
  if (state->force_status_error) {
    status_ec = std::error_code(EIO, std::generic_category());
  } else {
    status_ec.clear();
  }
  return state->is_regular_file_result;
}

struct ScopedResolvePythonExecutableOverrides {
  explicit ScopedResolvePythonExecutableOverrides(
      ResolvePythonExecutableOverrideState& state) noexcept
  {
    resolve_python_executable_override_state() = &state;
    resolve_python_candidates_override_for_test() =
        resolve_python_candidates_override_stub;
    resolve_python_is_regular_file_override_for_test() =
        resolve_python_is_regular_file_override_stub;
  }

  ~ScopedResolvePythonExecutableOverrides()
  {
    resolve_python_is_regular_file_override_for_test() = nullptr;
    resolve_python_candidates_override_for_test() = nullptr;
    resolve_python_executable_override_state() = nullptr;
  }
};

struct WaitForSignalNotificationReadOverrideState {
  std::vector<ssize_t> read_results;
  std::vector<int> errnos;
  std::size_t next_index = 0;
  int call_count = 0;
  int last_read_fd = -1;
  std::size_t last_buffer_size = 0;
};

auto
wait_for_signal_notification_read_override_state()
    -> WaitForSignalNotificationReadOverrideState*&
{
  static WaitForSignalNotificationReadOverrideState* state = nullptr;
  return state;
}

auto
wait_for_signal_notification_read_override_stub(
    int read_fd, void* /*buffer*/, std::size_t buffer_size) -> ssize_t
{
  auto* state = wait_for_signal_notification_read_override_state();
  if (state == nullptr) {
    return 0;
  }
  ++state->call_count;
  state->last_read_fd = read_fd;
  state->last_buffer_size = buffer_size;

  if (state->next_index >= state->read_results.size()) {
    errno = 0;
    return 0;
  }

  const auto index = state->next_index++;
  errno = index < state->errnos.size() ? state->errnos[index] : 0;
  return state->read_results[index];
}

struct ScopedWaitForSignalNotificationReadOverride {
  explicit ScopedWaitForSignalNotificationReadOverride(
      WaitForSignalNotificationReadOverrideState& state) noexcept
  {
    wait_for_signal_notification_read_override_state() = &state;
    wait_for_signal_notification_read_override_for_test() =
        wait_for_signal_notification_read_override_stub;
  }

  ~ScopedWaitForSignalNotificationReadOverride()
  {
    wait_for_signal_notification_read_override_for_test() = nullptr;
    wait_for_signal_notification_read_override_state() = nullptr;
  }
};

struct PlotScriptOverrideState {
  std::optional<std::filesystem::path> summary_path_result;
  std::optional<std::filesystem::path> locate_result;
  std::optional<int> run_result;
  int summary_path_calls = 0;
  int locate_calls = 0;
  int run_calls = 0;
  std::filesystem::path run_script_path;
  std::filesystem::path run_summary_path;
  std::filesystem::path run_output_path;
};

auto
plot_script_override_state() -> PlotScriptOverrideState*&
{
  static PlotScriptOverrideState* state = nullptr;
  return state;
}

auto
trace_summary_file_path_override_stub() -> std::optional<std::filesystem::path>
{
  auto* state = plot_script_override_state();
  if (state == nullptr) {
    return std::nullopt;
  }
  ++state->summary_path_calls;
  return state->summary_path_result;
}

auto
locate_plot_script_override_stub(const starpu_server::RuntimeConfig&)
    -> std::optional<std::filesystem::path>
{
  auto* state = plot_script_override_state();
  if (state == nullptr) {
    return std::nullopt;
  }
  ++state->locate_calls;
  return state->locate_result;
}

auto
run_plot_script_override_stub(
    const std::filesystem::path& script_path,
    const std::filesystem::path& summary_path,
    const std::filesystem::path& output_path) -> std::optional<int>
{
  auto* state = plot_script_override_state();
  if (state == nullptr) {
    return std::nullopt;
  }
  ++state->run_calls;
  state->run_script_path = script_path;
  state->run_summary_path = summary_path;
  state->run_output_path = output_path;
  return state->run_result;
}

struct ScopedPlotScriptOverrides {
  explicit ScopedPlotScriptOverrides(PlotScriptOverrideState& state) noexcept
  {
    plot_script_override_state() = &state;
    trace_summary_file_path_override_for_test() =
        trace_summary_file_path_override_stub;
    locate_plot_script_override_for_test() = locate_plot_script_override_stub;
    run_plot_script_override_for_test() = run_plot_script_override_stub;
  }

  ~ScopedPlotScriptOverrides()
  {
    run_plot_script_override_for_test() = nullptr;
    locate_plot_script_override_for_test() = nullptr;
    trace_summary_file_path_override_for_test() = nullptr;
    plot_script_override_state() = nullptr;
  }
};

using ModelPreparationTuple = std::tuple<
    torch::jit::script::Module, std::vector<torch::jit::script::Module>,
    std::vector<torch::Tensor>>;

struct PrepareModelsAndWarmupOverrideState {
  std::optional<ModelPreparationTuple> load_result;
  bool throw_from_warmup = false;
  std::string warmup_exception_message = "warmup failure";
  bool append_gpu_model_in_warmup = false;
  int load_calls = 0;
  int warmup_calls = 0;
  const starpu_server::RuntimeConfig* load_opts = nullptr;
  const starpu_server::RuntimeConfig* warmup_opts = nullptr;
  starpu_server::StarPUSetup* warmup_starpu = nullptr;
  size_t warmup_gpu_model_count = 0;
  size_t warmup_reference_output_count = 0;
};

auto
prepare_models_and_warmup_override_state()
    -> PrepareModelsAndWarmupOverrideState*&
{
  static PrepareModelsAndWarmupOverrideState* state = nullptr;
  return state;
}

auto
load_model_and_reference_output_override_stub(
    const starpu_server::RuntimeConfig& opts)
    -> std::optional<ModelPreparationTuple>
{
  auto* state = prepare_models_and_warmup_override_state();
  if (state == nullptr) {
    return std::nullopt;
  }
  ++state->load_calls;
  state->load_opts = &opts;
  return std::move(state->load_result);
}

void
run_warmup_override_stub(
    const starpu_server::RuntimeConfig& opts,
    starpu_server::StarPUSetup& starpu,
    torch::jit::script::Module& /*model_cpu*/,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<torch::Tensor>& reference_outputs)
{
  auto* state = prepare_models_and_warmup_override_state();
  if (state == nullptr) {
    return;
  }
  ++state->warmup_calls;
  state->warmup_opts = &opts;
  state->warmup_starpu = &starpu;
  state->warmup_gpu_model_count = models_gpu.size();
  state->warmup_reference_output_count = reference_outputs.size();
  if (state->append_gpu_model_in_warmup) {
    models_gpu.emplace_back("gpu_added_by_warmup");
  }
  if (state->throw_from_warmup) {
    throw std::runtime_error(state->warmup_exception_message);
  }
}

struct ScopedPrepareModelsAndWarmupOverrides {
  explicit ScopedPrepareModelsAndWarmupOverrides(
      PrepareModelsAndWarmupOverrideState& state) noexcept
  {
    prepare_models_and_warmup_override_state() = &state;
    load_model_and_reference_output_override_for_test() =
        load_model_and_reference_output_override_stub;
    run_warmup_override_for_test() = run_warmup_override_stub;
  }

  ~ScopedPrepareModelsAndWarmupOverrides()
  {
    run_warmup_override_for_test() = nullptr;
    load_model_and_reference_output_override_for_test() = nullptr;
    prepare_models_and_warmup_override_state() = nullptr;
  }
};

struct DescribeCpuAffinityOverrideState {
  hwloc_cpuset_t cpuset_to_return = nullptr;
  std::vector<int> cores;
  std::size_t next_index = 0;
  int provider_calls = 0;
  int first_calls = 0;
  int next_calls = 0;
  int free_calls = 0;
  int requested_worker_id = -1;
  hwloc_const_bitmap_t first_cpuset = nullptr;
  hwloc_const_bitmap_t next_cpuset = nullptr;
  hwloc_bitmap_t freed_cpuset = nullptr;
};

auto
describe_cpu_affinity_override_state() -> DescribeCpuAffinityOverrideState*&
{
  static DescribeCpuAffinityOverrideState* state = nullptr;
  return state;
}

auto
worker_cpuset_provider_override_stub(int worker_id) -> hwloc_cpuset_t
{
  auto* state = describe_cpu_affinity_override_state();
  if (state == nullptr) {
    return nullptr;
  }
  ++state->provider_calls;
  state->requested_worker_id = worker_id;
  state->next_index = 0;
  return state->cpuset_to_return;
}

auto
hwloc_bitmap_first_override_stub(hwloc_const_bitmap_t cpuset) -> int
{
  auto* state = describe_cpu_affinity_override_state();
  if (state == nullptr) {
    return -1;
  }
  ++state->first_calls;
  state->first_cpuset = cpuset;
  if (state->cores.empty()) {
    return -1;
  }
  state->next_index = 1;
  return state->cores.front();
}

auto
hwloc_bitmap_next_override_stub(
    hwloc_const_bitmap_t cpuset, int /*previous_core*/) -> int
{
  auto* state = describe_cpu_affinity_override_state();
  if (state == nullptr) {
    return -1;
  }
  ++state->next_calls;
  state->next_cpuset = cpuset;
  if (state->next_index >= state->cores.size()) {
    return -1;
  }
  return state->cores[state->next_index++];
}

void
hwloc_bitmap_free_override_stub(hwloc_bitmap_t cpuset)
{
  auto* state = describe_cpu_affinity_override_state();
  if (state == nullptr) {
    return;
  }
  ++state->free_calls;
  state->freed_cpuset = cpuset;
}

struct ScopedDescribeCpuAffinityOverrides {
  explicit ScopedDescribeCpuAffinityOverrides(
      DescribeCpuAffinityOverrideState& state) noexcept
  {
    describe_cpu_affinity_override_state() = &state;
    worker_cpuset_provider_override_for_test() =
        worker_cpuset_provider_override_stub;
    hwloc_bitmap_first_override_for_test() = hwloc_bitmap_first_override_stub;
    hwloc_bitmap_next_override_for_test() = hwloc_bitmap_next_override_stub;
    hwloc_bitmap_free_override_for_test() = hwloc_bitmap_free_override_stub;
  }

  ~ScopedDescribeCpuAffinityOverrides()
  {
    hwloc_bitmap_free_override_for_test() = nullptr;
    hwloc_bitmap_next_override_for_test() = nullptr;
    hwloc_bitmap_first_override_for_test() = nullptr;
    worker_cpuset_provider_override_for_test() = nullptr;
    describe_cpu_affinity_override_state() = nullptr;
  }
};

struct LogWorkerInventoryOverrideState {
  int worker_count = 0;
  std::vector<enum starpu_worker_archtype> worker_types;
  std::vector<int> device_ids;
  std::vector<std::string> cpu_affinities;
  int worker_count_calls = 0;
  int worker_type_calls = 0;
  int worker_device_id_calls = 0;
  int describe_cpu_affinity_calls = 0;
  std::vector<int> requested_worker_ids_for_type;
  std::vector<int> requested_worker_ids_for_device;
  std::vector<int> requested_worker_ids_for_affinity;
};

auto
log_worker_inventory_override_state() -> LogWorkerInventoryOverrideState*&
{
  static LogWorkerInventoryOverrideState* state = nullptr;
  return state;
}

auto
worker_count_override_stub() -> decltype(starpu_worker_get_count())
{
  auto* state = log_worker_inventory_override_state();
  if (state == nullptr) {
    return 0;
  }
  ++state->worker_count_calls;
  return static_cast<decltype(starpu_worker_get_count())>(state->worker_count);
}

auto
worker_type_override_stub(int worker_id)
    -> decltype(starpu_worker_get_type(worker_id))
{
  auto* state = log_worker_inventory_override_state();
  if (state == nullptr) {
    return STARPU_CPU_WORKER;
  }
  ++state->worker_type_calls;
  state->requested_worker_ids_for_type.push_back(worker_id);
  if (worker_id < 0 ||
      static_cast<std::size_t>(worker_id) >= state->worker_types.size()) {
    return STARPU_CPU_WORKER;
  }
  return state->worker_types[static_cast<std::size_t>(worker_id)];
}

auto
worker_device_id_override_stub(int worker_id)
    -> decltype(starpu_worker_get_devid(worker_id))
{
  auto* state = log_worker_inventory_override_state();
  if (state == nullptr) {
    return -1;
  }
  ++state->worker_device_id_calls;
  state->requested_worker_ids_for_device.push_back(worker_id);
  if (worker_id < 0 ||
      static_cast<std::size_t>(worker_id) >= state->device_ids.size()) {
    return -1;
  }
  return state->device_ids[static_cast<std::size_t>(worker_id)];
}

auto
describe_cpu_affinity_override_stub(int worker_id) -> std::string
{
  auto* state = log_worker_inventory_override_state();
  if (state == nullptr) {
    return {};
  }
  ++state->describe_cpu_affinity_calls;
  state->requested_worker_ids_for_affinity.push_back(worker_id);
  if (worker_id < 0 ||
      static_cast<std::size_t>(worker_id) >= state->cpu_affinities.size()) {
    return {};
  }
  return state->cpu_affinities[static_cast<std::size_t>(worker_id)];
}

struct ScopedLogWorkerInventoryOverrides {
  explicit ScopedLogWorkerInventoryOverrides(
      LogWorkerInventoryOverrideState& state) noexcept
  {
    log_worker_inventory_override_state() = &state;
    worker_count_override_for_test() = worker_count_override_stub;
    worker_type_override_for_test() = worker_type_override_stub;
    worker_device_id_override_for_test() = worker_device_id_override_stub;
    describe_cpu_affinity_override_for_test() =
        describe_cpu_affinity_override_stub;
  }

  ~ScopedLogWorkerInventoryOverrides()
  {
    describe_cpu_affinity_override_for_test() = nullptr;
    worker_device_id_override_for_test() = nullptr;
    worker_type_override_for_test() = nullptr;
    worker_count_override_for_test() = nullptr;
    log_worker_inventory_override_state() = nullptr;
  }
};

struct OccupiedLoopbackPort {
  int socket_fd = -1;
  int port = -1;

  OccupiedLoopbackPort() = default;
  OccupiedLoopbackPort(const OccupiedLoopbackPort&) = delete;
  auto operator=(const OccupiedLoopbackPort&) -> OccupiedLoopbackPort& = delete;
  OccupiedLoopbackPort(OccupiedLoopbackPort&& other) noexcept
      : socket_fd(other.socket_fd), port(other.port)
  {
    other.socket_fd = -1;
    other.port = -1;
  }
  auto operator=(OccupiedLoopbackPort&& other) noexcept -> OccupiedLoopbackPort&
  {
    if (this == &other) {
      return *this;
    }
    if (socket_fd >= 0) {
      ::close(socket_fd);
    }
    socket_fd = other.socket_fd;
    port = other.port;
    other.socket_fd = -1;
    other.port = -1;
    return *this;
  }
  ~OccupiedLoopbackPort()
  {
    if (socket_fd >= 0) {
      ::close(socket_fd);
    }
  }
};

auto
fixture_model_path() -> std::filesystem::path
{
  const auto relative = std::filesystem::path(__FILE__).parent_path() / ".." /
                        ".." / "e2e" / "fixtures" / "simple_model.ts";
  std::error_code ec;
  const auto canonical = std::filesystem::weakly_canonical(relative, ec);
  if (!ec) {
    return canonical;
  }
  return relative;
}

auto
write_temp_config_file() -> TempFileGuard
{
  static std::atomic<std::uint64_t> temp_file_counter{0};
  const auto suffix = temp_file_counter.fetch_add(1, std::memory_order_relaxed);
  TempFileGuard guard{
      std::filesystem::temp_directory_path() /
      ("server_main_test_" + std::to_string(suffix) + ".yml")};

  std::ofstream cfg(guard.path);
  EXPECT_TRUE(cfg.is_open());

  const auto model_path = fixture_model_path();
  cfg << "name: unit_server_main\n";
  cfg << "model: " << model_path.string() << "\n";
  cfg << "inputs:\n";
  cfg << "  - { name: input0, data_type: TYPE_FP32, dims: [1, 2] }\n";
  cfg << "outputs:\n";
  cfg << "  - { name: output0, data_type: TYPE_FP32, dims: [1, 2] }\n";
  cfg << "pool_size: 1\n";
  cfg << "max_batch_size: 1\n";
  cfg << "batch_coalesce_timeout_ms: 0\n";
  cfg.close();

  EXPECT_TRUE(std::filesystem::exists(guard.path));
  return guard;
}

auto
pick_unused_port() -> int
{
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;

  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return -1;
  }

  socklen_t addr_len = sizeof(addr);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &addr_len) != 0) {
    ::close(fd);
    return -1;
  }

  const int port = ntohs(addr.sin_port);
  ::close(fd);
  return port;
}

auto
occupy_loopback_port() -> OccupiedLoopbackPort
{
  OccupiedLoopbackPort occupied;
  occupied.socket_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (occupied.socket_fd < 0) {
    return occupied;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  if (::bind(
          occupied.socket_fd, reinterpret_cast<sockaddr*>(&addr),
          sizeof(addr)) != 0) {
    ::close(occupied.socket_fd);
    occupied.socket_fd = -1;
    return occupied;
  }
  if (::listen(occupied.socket_fd, 1) != 0) {
    ::close(occupied.socket_fd);
    occupied.socket_fd = -1;
    return occupied;
  }

  socklen_t addr_len = sizeof(addr);
  if (::getsockname(
          occupied.socket_fd, reinterpret_cast<sockaddr*>(&addr), &addr_len) !=
      0) {
    ::close(occupied.socket_fd);
    occupied.socket_fd = -1;
    return occupied;
  }
  occupied.port = ntohs(addr.sin_port);
  return occupied;
}

auto
wait_for_channel_ready(
    const std::string& address,
    std::chrono::system_clock::duration timeout) -> bool
{
  auto channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  return channel->WaitForConnected(std::chrono::system_clock::now() + timeout);
}

auto
make_temp_test_path(const std::string& stem, const std::string& extension)
    -> std::filesystem::path
{
  static std::atomic<std::uint64_t> temp_artifact_counter{0};
  const auto id = temp_artifact_counter.fetch_add(1, std::memory_order_relaxed);
  return std::filesystem::temp_directory_path() /
         (stem + "_" + std::to_string(id) + extension);
}

auto
make_runtime_config_for_starpu_setup() -> starpu_server::RuntimeConfig
{
  starpu_server::RuntimeConfig opts;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 8;
  return opts;
}

auto
spawn_exiting_child(int exit_code) -> pid_t
{
  const pid_t pid = ::fork();
  if (pid == 0) {
    _exit(exit_code);
  }
  return pid;
}

auto
spawn_sleeping_child(bool ignore_sigterm) -> pid_t
{
  int ready_pipe[2]{-1, -1};
  if (::pipe(ready_pipe) != 0) {
    return -1;
  }

  const pid_t pid = ::fork();
  if (pid == 0) {
    ::close(ready_pipe[0]);
    if (ignore_sigterm) {
      sigset_t blocked{};
      ::sigemptyset(&blocked);
      ::sigaddset(&blocked, SIGTERM);
      (void)::sigprocmask(SIG_BLOCK, &blocked, nullptr);
    }
    const char ready = '1';
    (void)::write(ready_pipe[1], &ready, 1);
    ::close(ready_pipe[1]);
    while (true) {
      ::pause();
    }
  }
  ::close(ready_pipe[1]);
  if (pid < 0) {
    ::close(ready_pipe[0]);
    return -1;
  }
  char ready = 0;
  const auto bytes_read = ::read(ready_pipe[0], &ready, 1);
  ::close(ready_pipe[0]);
  if (bytes_read != 1) {
    (void)::kill(pid, SIGKILL);
    (void)wait_for_exit_blocking(pid);
    return -1;
  }
  return pid;
}

TEST(ServerMainArgs, HandleProgramArgumentsParsesLongConfigFlag)
{
  auto config_guard = write_temp_config_file();

  const std::string arg0 = "starpu_server";
  const std::string arg1 = "--config";
  const std::string arg2 = config_guard.path.string();
  const std::array<const char*, 3> argv{
      arg0.c_str(), arg1.c_str(), arg2.c_str()};

  auto cfg = handle_program_arguments({argv.data(), argv.size()});

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.name, "unit_server_main");
  ASSERT_TRUE(cfg.model.has_value());
  EXPECT_EQ(cfg.model->inputs.size(), 1U);
  EXPECT_EQ(cfg.model->outputs.size(), 1U);
  EXPECT_EQ(cfg.batching.max_batch_size, 1);
}

TEST(ServerMainArgs, HandleProgramArgumentsParsesShortConfigFlag)
{
  auto config_guard = write_temp_config_file();

  const std::string arg0 = "starpu_server";
  const std::string arg1 = "-c";
  const std::string arg2 = config_guard.path.string();
  const std::array<const char*, 3> argv{
      arg0.c_str(), arg1.c_str(), arg2.c_str()};

  auto cfg = handle_program_arguments({argv.data(), argv.size()});

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.name, "unit_server_main");
}

TEST(ServerMainSignal, SignalHandlerSetsStopFlag)
{
  signal_stop_requested_flag() = 0;
  signal_handler(SIGINT);
  EXPECT_EQ(signal_stop_requested_flag(), 1);
  signal_stop_requested_flag() = 0;
}

TEST(ServerMainThreadExceptionState, CaptureStoresFirstExceptionAndThreadName)
{
  ThreadExceptionState state;
  state.capture(
      "grpc-server",
      std::make_exception_ptr(std::runtime_error("grpc thread failure")));

  auto [captured_exception, thread_name] = state.take();
  ASSERT_NE(captured_exception, nullptr);
  EXPECT_EQ(thread_name, "grpc-server");

  try {
    std::rethrow_exception(captured_exception);
    FAIL() << "Expected runtime_error to be rethrown.";
  }
  catch (const std::runtime_error& error) {
    EXPECT_EQ(std::string(error.what()), "grpc thread failure");
  }
  catch (...) {
    FAIL() << "Expected std::runtime_error.";
  }
}

TEST(
    ServerMainThreadExceptionState,
    CaptureKeepsFirstExceptionWhenCalledMultipleTimes)
{
  ThreadExceptionState state;
  state.capture(
      "signal-notifier",
      std::make_exception_ptr(std::logic_error("first failure")));
  state.capture(
      "starpu-worker",
      std::make_exception_ptr(std::runtime_error("second failure")));

  auto [captured_exception, thread_name] = state.take();
  ASSERT_NE(captured_exception, nullptr);
  EXPECT_EQ(thread_name, "signal-notifier");

  try {
    std::rethrow_exception(captured_exception);
    FAIL() << "Expected logic_error to be rethrown.";
  }
  catch (const std::logic_error& error) {
    EXPECT_EQ(std::string(error.what()), "first failure");
  }
  catch (...) {
    FAIL() << "Expected std::logic_error.";
  }
}

TEST(
    ServerMainModelPreparation,
    PrepareModelsAndWarmupThrowsModelLoadingExceptionWhenLoadingFails)
{
  auto opts = make_runtime_config_for_starpu_setup();
  starpu_server::StarPUSetup starpu(opts);

  PrepareModelsAndWarmupOverrideState override_state;
  ScopedPrepareModelsAndWarmupOverrides overrides(override_state);

  try {
    (void)prepare_models_and_warmup(opts, starpu);
    FAIL() << "Expected ModelLoadingException.";
  }
  catch (const starpu_server::ModelLoadingException& error) {
    EXPECT_EQ(
        std::string(error.what()), "Failed to load model or reference outputs");
  }
  catch (...) {
    FAIL() << "Expected starpu_server::ModelLoadingException.";
  }

  EXPECT_EQ(override_state.load_calls, 1);
  EXPECT_EQ(override_state.warmup_calls, 0);
  EXPECT_EQ(override_state.load_opts, &opts);
}

TEST(
    ServerMainModelPreparation,
    PrepareModelsAndWarmupRunsWarmupAndReturnsModels)
{
  auto opts = make_runtime_config_for_starpu_setup();
  starpu_server::StarPUSetup starpu(opts);

  PrepareModelsAndWarmupOverrideState override_state;
  override_state.load_result = ModelPreparationTuple{
      torch::jit::script::Module("cpu_model"),
      std::vector<torch::jit::script::Module>{
          torch::jit::script::Module("gpu_model_0")},
      std::vector<torch::Tensor>{
          torch::ones({2, 2}, torch::dtype(torch::kFloat32))}};
  override_state.append_gpu_model_in_warmup = true;
  ScopedPrepareModelsAndWarmupOverrides overrides(override_state);

  auto [model_cpu, models_gpu, reference_outputs] =
      prepare_models_and_warmup(opts, starpu);

  (void)model_cpu;
  EXPECT_EQ(override_state.load_calls, 1);
  EXPECT_EQ(override_state.warmup_calls, 1);
  EXPECT_EQ(override_state.load_opts, &opts);
  EXPECT_EQ(override_state.warmup_opts, &opts);
  EXPECT_EQ(override_state.warmup_starpu, &starpu);
  EXPECT_EQ(override_state.warmup_gpu_model_count, 1U);
  EXPECT_EQ(override_state.warmup_reference_output_count, 1U);
  EXPECT_EQ(models_gpu.size(), 2U);
  ASSERT_EQ(reference_outputs.size(), 1U);
  EXPECT_TRUE(torch::equal(
      reference_outputs[0],
      torch::ones({2, 2}, torch::dtype(torch::kFloat32))));
}

TEST(
    ServerMainModelPreparation, PrepareModelsAndWarmupPropagatesWarmupException)
{
  auto opts = make_runtime_config_for_starpu_setup();
  starpu_server::StarPUSetup starpu(opts);

  PrepareModelsAndWarmupOverrideState override_state;
  override_state.load_result = ModelPreparationTuple{
      torch::jit::script::Module("cpu_model"),
      std::vector<torch::jit::script::Module>{}, std::vector<torch::Tensor>{}};
  override_state.throw_from_warmup = true;
  override_state.warmup_exception_message = "forced warmup failure";
  ScopedPrepareModelsAndWarmupOverrides overrides(override_state);

  try {
    (void)prepare_models_and_warmup(opts, starpu);
    FAIL() << "Expected runtime_error from warmup.";
  }
  catch (const std::runtime_error& error) {
    EXPECT_EQ(std::string(error.what()), "forced warmup failure");
  }
  catch (...) {
    FAIL() << "Expected std::runtime_error.";
  }

  EXPECT_EQ(override_state.load_calls, 1);
  EXPECT_EQ(override_state.warmup_calls, 1);
}

TEST(ServerMainModelNameResolution, UsesModelNameWhenConfiguredAndNonEmpty)
{
  starpu_server::RuntimeConfig opts;
  opts.name = "fallback_name";
  opts.model = starpu_server::ModelConfig{};
  opts.model->name = "configured_model_name";

  EXPECT_EQ(resolve_default_model_name(opts), "configured_model_name");
}

TEST(ServerMainModelNameResolution, FallsBackToConfigNameWhenModelNameIsEmpty)
{
  starpu_server::RuntimeConfig opts;
  opts.name = "fallback_name";
  opts.model = starpu_server::ModelConfig{};
  opts.model->name = "";

  EXPECT_EQ(resolve_default_model_name(opts), "fallback_name");
}

TEST(ServerMainModelNameResolution, FallsBackToConfigNameWhenModelIsNotSet)
{
  starpu_server::RuntimeConfig opts;
  opts.name = "fallback_name";
  opts.model.reset();

  EXPECT_EQ(resolve_default_model_name(opts), "fallback_name");
}

TEST(ServerMainWorkerTypeLabel, ReturnsCpuLabelForCpuWorker)
{
  EXPECT_EQ(worker_type_label(STARPU_CPU_WORKER), "CPU");
}

TEST(ServerMainWorkerTypeLabel, ReturnsCudaLabelForCudaWorker)
{
  EXPECT_EQ(worker_type_label(STARPU_CUDA_WORKER), "CUDA");
}

TEST(ServerMainWorkerTypeLabel, ReturnsOtherLabelForUnknownWorkerType)
{
  constexpr auto unknown_worker_type =
      static_cast<enum starpu_worker_archtype>(-12345);
  EXPECT_EQ(worker_type_label(unknown_worker_type), "Other(-12345)");
}

TEST(ServerMainCpuCoreRanges, ReturnsEmptyStringForEmptyInput)
{
  EXPECT_EQ(format_cpu_core_ranges({}), "");
}

TEST(ServerMainCpuCoreRanges, FormatsSingleCpuAsSingleValue)
{
  EXPECT_EQ(format_cpu_core_ranges({7}), "7");
}

TEST(ServerMainCpuCoreRanges, FormatsMixedContiguousAndSingleRanges)
{
  EXPECT_EQ(format_cpu_core_ranges({1, 2, 3, 5, 7, 8}), "1-3,5,7-8");
}

TEST(ServerMainCpuCoreRanges, FormatsMultipleSingleValuesWithCommas)
{
  EXPECT_EQ(format_cpu_core_ranges({2, 4, 6}), "2,4,6");
}

TEST(ServerMainCpuAffinity, ReturnsEmptyStringWhenCpusetIsNull)
{
  DescribeCpuAffinityOverrideState override_state;
  override_state.cpuset_to_return = nullptr;
  ScopedDescribeCpuAffinityOverrides overrides(override_state);

  EXPECT_EQ(describe_cpu_affinity(11), "");
  EXPECT_EQ(override_state.provider_calls, 1);
  EXPECT_EQ(override_state.requested_worker_id, 11);
  EXPECT_EQ(override_state.first_calls, 0);
  EXPECT_EQ(override_state.next_calls, 0);
  EXPECT_EQ(override_state.free_calls, 0);
}

TEST(ServerMainCpuAffinity, ReturnsEmptyStringWhenCpusetContainsNoCore)
{
  DescribeCpuAffinityOverrideState override_state;
  override_state.cpuset_to_return =
      reinterpret_cast<hwloc_cpuset_t>(static_cast<std::uintptr_t>(0x1));
  override_state.cores = {};
  ScopedDescribeCpuAffinityOverrides overrides(override_state);

  EXPECT_EQ(describe_cpu_affinity(3), "");
  EXPECT_EQ(override_state.provider_calls, 1);
  EXPECT_EQ(override_state.requested_worker_id, 3);
  EXPECT_EQ(override_state.first_calls, 1);
  EXPECT_EQ(override_state.next_calls, 0);
  EXPECT_EQ(override_state.free_calls, 1);
  EXPECT_EQ(override_state.first_cpuset, override_state.cpuset_to_return);
  EXPECT_EQ(override_state.freed_cpuset, override_state.cpuset_to_return);
}

TEST(ServerMainCpuAffinity, FormatsCoresAndFreesCpusetWhenCoresArePresent)
{
  DescribeCpuAffinityOverrideState override_state;
  override_state.cpuset_to_return =
      reinterpret_cast<hwloc_cpuset_t>(static_cast<std::uintptr_t>(0x2));
  override_state.cores = {0, 1, 2, 4, 6, 7};
  ScopedDescribeCpuAffinityOverrides overrides(override_state);

  EXPECT_EQ(describe_cpu_affinity(5), "0-2,4,6-7");
  EXPECT_EQ(override_state.provider_calls, 1);
  EXPECT_EQ(override_state.requested_worker_id, 5);
  EXPECT_EQ(override_state.first_calls, 1);
  EXPECT_EQ(override_state.next_calls, 6);
  EXPECT_EQ(override_state.free_calls, 1);
  EXPECT_EQ(override_state.first_cpuset, override_state.cpuset_to_return);
  EXPECT_EQ(override_state.next_cpuset, override_state.cpuset_to_return);
  EXPECT_EQ(override_state.freed_cpuset, override_state.cpuset_to_return);
}

TEST(ServerMainWorkerInventory, LogsOnlyHeaderWhenNoWorkersAreConfigured)
{
  LogWorkerInventoryOverrideState override_state;
  override_state.worker_count = 0;
  ScopedLogWorkerInventoryOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.verbosity = starpu_server::VerbosityLevel::Info;

  OStreamCapture capture_out(std::cout);
  log_worker_inventory(opts);

  EXPECT_EQ(
      capture_out.str(), expected_info_log("Configured 0 StarPU worker(s)."));
  EXPECT_EQ(override_state.worker_count_calls, 1);
  EXPECT_EQ(override_state.worker_type_calls, 0);
  EXPECT_EQ(override_state.worker_device_id_calls, 0);
  EXPECT_EQ(override_state.describe_cpu_affinity_calls, 0);
}

TEST(
    ServerMainWorkerInventory,
    LogsPerWorkerTypeDeviceAndCpuAffinityForMixedWorkers)
{
  LogWorkerInventoryOverrideState override_state;
  override_state.worker_count = 3;
  override_state.worker_types = {
      STARPU_CPU_WORKER, STARPU_CPU_WORKER, STARPU_CUDA_WORKER};
  override_state.device_ids = {0, -1, 2};
  override_state.cpu_affinities = {"0-3", "", "should_not_be_used"};
  ScopedLogWorkerInventoryOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.verbosity = starpu_server::VerbosityLevel::Info;

  OStreamCapture capture_out(std::cout);
  log_worker_inventory(opts);

  const auto expected =
      expected_info_log("Configured 3 StarPU worker(s).") +
      expected_info_log("Worker  0: type=CPU, device id=0, cores=0-3") +
      expected_info_log("Worker  1: type=CPU, device id=N/A") +
      expected_info_log("Worker  2: type=CUDA, device id=2");
  EXPECT_EQ(capture_out.str(), expected);
  EXPECT_EQ(override_state.worker_count_calls, 1);
  EXPECT_EQ(override_state.worker_type_calls, 3);
  EXPECT_EQ(override_state.worker_device_id_calls, 3);
  EXPECT_EQ(override_state.describe_cpu_affinity_calls, 2);
  EXPECT_EQ(
      override_state.requested_worker_ids_for_type,
      std::vector<int>({0, 1, 2}));
  EXPECT_EQ(
      override_state.requested_worker_ids_for_device,
      std::vector<int>({0, 1, 2}));
  EXPECT_EQ(
      override_state.requested_worker_ids_for_affinity,
      std::vector<int>({0, 1}));
}

static_assert(std::is_nothrow_destructible_v<RuntimeCleanupGuard>);
static_assert(std::is_nothrow_invocable_v<
              decltype(&RuntimeCleanupGuard::Dismiss), RuntimeCleanupGuard&>);
static_assert(std::is_nothrow_invocable_v<
              decltype(&RuntimeCleanupGuard::ResetTraceLoggerNoexceptForTest)>);
static_assert(std::is_nothrow_invocable_v<
              decltype(&RuntimeCleanupGuard::ShutdownMetricsNoexceptForTest)>);
static_assert(std::is_nothrow_invocable_v<
              decltype(&SignalNotificationPipe::SetPipeFailureForTest), bool>);
static_assert(
    std::is_nothrow_invocable_v<
        decltype(&SignalNotificationPipe::SetSetNonBlockingFailureForTest),
        bool>);
static_assert(std::is_nothrow_invocable_r_v<
              WaitForSignalNotificationReadOverrideForTestFn&,
              decltype(wait_for_signal_notification_read_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              ResolvePythonCandidatesOverrideForTestFn&,
              decltype(resolve_python_candidates_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              ResolvePythonIsRegularFileOverrideForTestFn&,
              decltype(resolve_python_is_regular_file_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              RunPlotScriptOverrideForTestFn&,
              decltype(run_plot_script_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              LocatePlotScriptOverrideForTestFn&,
              decltype(locate_plot_script_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              TraceSummaryFilePathOverrideForTestFn&,
              decltype(trace_summary_file_path_override_for_test)>);
static_assert(
    std::is_nothrow_invocable_r_v<
        std::remove_reference_t<
            decltype(load_model_and_reference_output_override_for_test())>&,
        decltype(load_model_and_reference_output_override_for_test)>);
static_assert(
    std::is_nothrow_invocable_r_v<
        std::remove_reference_t<decltype(run_warmup_override_for_test())>&,
        decltype(run_warmup_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              WorkerCpusetProviderOverrideForTestFn&,
              decltype(worker_cpuset_provider_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              HwlocBitmapFirstOverrideForTestFn&,
              decltype(hwloc_bitmap_first_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              HwlocBitmapNextOverrideForTestFn&,
              decltype(hwloc_bitmap_next_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              HwlocBitmapFreeOverrideForTestFn&,
              decltype(hwloc_bitmap_free_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              WorkerCountOverrideForTestFn&,
              decltype(worker_count_override_for_test)>);
static_assert(
    std::is_nothrow_invocable_r_v<
        WorkerTypeOverrideForTestFn&, decltype(worker_type_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              WorkerDeviceIdOverrideForTestFn&,
              decltype(worker_device_id_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              DescribeCpuAffinityOverrideForTestFn&,
              decltype(describe_cpu_affinity_override_for_test)>);

TEST(ServerMainRuntimeCleanupGuard, DestructorPerformsCleanupWhenActive)
{
  auto& tracer = starpu_server::BatchingTraceLogger::instance();
  tracer.configure(false, "");
  starpu_server::shutdown_metrics();

  tracer.configure(true, "");
  ASSERT_TRUE(tracer.enabled());
  ASSERT_TRUE(starpu_server::init_metrics(0));
  ASSERT_NE(starpu_server::get_metrics(), nullptr);

  {
    RuntimeCleanupGuard guard;
  }

  EXPECT_FALSE(tracer.enabled());
  EXPECT_EQ(starpu_server::get_metrics(), nullptr);
}

TEST(ServerMainRuntimeCleanupGuard, DestructorSkipsCleanupAfterDismiss)
{
  auto& tracer = starpu_server::BatchingTraceLogger::instance();
  tracer.configure(false, "");
  starpu_server::shutdown_metrics();

  tracer.configure(true, "");
  ASSERT_TRUE(tracer.enabled());
  ASSERT_TRUE(starpu_server::init_metrics(0));
  ASSERT_NE(starpu_server::get_metrics(), nullptr);

  {
    RuntimeCleanupGuard guard;
    EXPECT_NO_THROW(guard.Dismiss());
    EXPECT_NO_THROW(guard.Dismiss());
  }

  EXPECT_TRUE(tracer.enabled());
  EXPECT_NE(starpu_server::get_metrics(), nullptr);

  tracer.configure(false, "");
  starpu_server::shutdown_metrics();
}

TEST(ServerMainRuntimeCleanupGuard, ResetTraceLoggerNoexceptDisablesTracer)
{
  auto& tracer = starpu_server::BatchingTraceLogger::instance();
  tracer.configure(false, "");
  tracer.configure(true, "");
  ASSERT_TRUE(tracer.enabled());

  EXPECT_NO_THROW(RuntimeCleanupGuard::ResetTraceLoggerNoexceptForTest());
  EXPECT_FALSE(tracer.enabled());
}

TEST(
    ServerMainRuntimeCleanupGuard,
    ResetTraceLoggerNoexceptSwallowsInternalExceptions)
{
  auto& tracer = starpu_server::BatchingTraceLogger::instance();
  tracer.configure(false, "");
  tracer.configure(true, "");
  ASSERT_TRUE(tracer.enabled());

  ScopedResetTraceLoggerForcedThrow forced_throw;
  EXPECT_NO_THROW(RuntimeCleanupGuard::ResetTraceLoggerNoexceptForTest());

  EXPECT_TRUE(tracer.enabled());
  tracer.configure(false, "");
}

TEST(ServerMainRuntimeCleanupGuard, ShutdownMetricsNoexceptShutsDownRegistry)
{
  starpu_server::shutdown_metrics();
  ASSERT_TRUE(starpu_server::init_metrics(0));
  ASSERT_NE(starpu_server::get_metrics(), nullptr);

  EXPECT_NO_THROW(RuntimeCleanupGuard::ShutdownMetricsNoexceptForTest());
  EXPECT_EQ(starpu_server::get_metrics(), nullptr);
}

TEST(
    ServerMainRuntimeCleanupGuard,
    ShutdownMetricsNoexceptSwallowsInternalExceptions)
{
  starpu_server::shutdown_metrics();
  ASSERT_TRUE(starpu_server::init_metrics(0));
  ASSERT_NE(starpu_server::get_metrics(), nullptr);

  ScopedShutdownMetricsForcedThrow forced_throw;
  EXPECT_NO_THROW(RuntimeCleanupGuard::ShutdownMetricsNoexceptForTest());

  EXPECT_NE(starpu_server::get_metrics(), nullptr);
  starpu_server::shutdown_metrics();
}

TEST(
    ServerMainSignalNotificationPipe, ConstructorFallsBackWhenPipeCreationFails)
{
  signal_stop_notify_fd() = -1;
  ScopedSignalPipeForcedPipeFailure forced_failure;
  SignalNotificationPipe signal_pipe;

  EXPECT_FALSE(signal_pipe.active());
  EXPECT_EQ(signal_pipe.read_fd(), -1);
  EXPECT_EQ(signal_stop_notify_fd(), -1);
}

TEST(
    ServerMainSignalNotificationPipe,
    ConstructorFallsBackWhenSetNonBlockingFails)
{
  signal_stop_notify_fd() = -1;
  ScopedSignalPipeForcedSetNonBlockingFailure forced_failure;
  SignalNotificationPipe signal_pipe;

  EXPECT_FALSE(signal_pipe.active());
  EXPECT_EQ(signal_pipe.read_fd(), -1);
  EXPECT_EQ(signal_stop_notify_fd(), -1);
}

TEST(ServerMainSignalNotificationPipe, SetNonBlockingReturnsFalseForNegativeFd)
{
  EXPECT_FALSE(SignalNotificationPipe::SetNonBlockingForTest(-1));
}

TEST(
    ServerMainSignalNotificationPipe,
    SetNonBlockingReturnsFalseWhenFcntlGetFlFails)
{
  std::array<int, 2> file_descriptors{-1, -1};
  ASSERT_EQ(::pipe(file_descriptors.data()), 0);

  const int closed_fd = file_descriptors[1];
  ASSERT_EQ(::close(file_descriptors[1]), 0);
  file_descriptors[1] = -1;

  EXPECT_FALSE(SignalNotificationPipe::SetNonBlockingForTest(closed_fd));

  ASSERT_EQ(::close(file_descriptors[0]), 0);
}

TEST(ServerMainSignalNotificationWait, ReturnsImmediatelyWhenReadFdIsNegative)
{
  WaitForSignalNotificationReadOverrideState override_state;
  override_state.read_results = {-1};
  override_state.errnos = {EIO};
  ScopedWaitForSignalNotificationReadOverride override_guard(override_state);

  OStreamCapture capture_err(std::cerr);
  wait_for_signal_notification(-1);

  EXPECT_EQ(override_state.call_count, 0);
  EXPECT_TRUE(capture_err.str().empty());
}

TEST(ServerMainSignalNotificationWait, RetriesReadWhenInterruptedBySignal)
{
  WaitForSignalNotificationReadOverrideState override_state;
  override_state.read_results = {-1, 0};
  override_state.errnos = {EINTR, 0};
  ScopedWaitForSignalNotificationReadOverride override_guard(override_state);

  OStreamCapture capture_err(std::cerr);
  wait_for_signal_notification(42);

  EXPECT_EQ(override_state.call_count, 2);
  EXPECT_EQ(override_state.last_read_fd, 42);
  EXPECT_EQ(override_state.last_buffer_size, static_cast<std::size_t>(16));
  EXPECT_TRUE(capture_err.str().empty());
}

TEST(ServerMainSignalNotificationWait, LogsWarningWhenReadFailsWithNonEintr)
{
  WaitForSignalNotificationReadOverrideState override_state;
  override_state.read_results = {-1};
  override_state.errnos = {EIO};
  ScopedWaitForSignalNotificationReadOverride override_guard(override_state);

  OStreamCapture capture_err(std::cerr);
  wait_for_signal_notification(7);

  EXPECT_EQ(override_state.call_count, 1);
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log(
          std::string("Failed while waiting for stop signal notification: ") +
          std::strerror(EIO)));
}

TEST(ServerMainOrchestration, LaunchThreadsStopsAfterSignal)
{
  if (running_under_tsan()) {
    GTEST_SKIP()
        << "Skipped under TSAN: gRPC event-engine triggers known external race "
           "in absl::raw_hash_set";
  }

  constexpr auto kLaunchTimeout = std::chrono::seconds(10);
  constexpr auto kServerStartTimeout = std::chrono::seconds(5);

  const int port = pick_unused_port();
  ASSERT_GT(port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(port);

  starpu_server::RuntimeConfig opts;
  opts.server_address = address;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 8;

  signal_stop_requested_flag() = 0;
  auto& ctx = server_context();
  ctx.stop_requested.store(false, std::memory_order_relaxed);

  starpu_server::StarPUSetup starpu(opts);
  torch::jit::script::Module model_cpu("m");
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> reference_outputs;
  starpu_server::InferenceQueue queue(opts.batching.max_queue_size);

  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();

  std::jthread launch_thread([&]() {
    try {
      launch_threads(
          opts, starpu, model_cpu, models_gpu, reference_outputs, queue);
      done_promise.set_value();
    }
    catch (...) {
      done_promise.set_exception(std::current_exception());
    }
  });

  ASSERT_TRUE(wait_for_channel_ready(address, kServerStartTimeout))
      << "Timed out waiting for gRPC server startup";
  signal_handler(SIGTERM);

  if (done_future.wait_for(kLaunchTimeout) != std::future_status::ready) {
    signal_stop_requested_flag() = 1;
    ctx.stop_requested.store(true, std::memory_order_relaxed);
    ctx.stop_cv.notify_one();
  }

  ASSERT_EQ(done_future.wait_for(kLaunchTimeout), std::future_status::ready)
      << "launch_threads did not stop within timeout";
  EXPECT_NO_THROW(done_future.get());

  signal_stop_requested_flag() = 0;
}

TEST(
    ServerMainOrchestration,
    LaunchThreadsStopsWhenGrpcStartupFailsOnOccupiedPort)
{
  if (running_under_tsan()) {
    GTEST_SKIP()
        << "Skipped under TSAN: gRPC event-engine triggers known external race "
           "in absl::raw_hash_set";
  }

  constexpr auto kLaunchTimeout = std::chrono::seconds(10);

  auto occupied_port = occupy_loopback_port();
  ASSERT_GE(occupied_port.socket_fd, 0);
  ASSERT_GT(occupied_port.port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(occupied_port.port);

  starpu_server::RuntimeConfig opts;
  opts.server_address = address;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 8;

  signal_stop_requested_flag() = 0;
  auto& ctx = server_context();
  ctx.stop_requested.store(false, std::memory_order_relaxed);

  starpu_server::StarPUSetup starpu(opts);
  torch::jit::script::Module model_cpu("m");
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> reference_outputs;
  starpu_server::InferenceQueue queue(opts.batching.max_queue_size);

  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();
  std::jthread launch_thread([&]() {
    try {
      launch_threads(
          opts, starpu, model_cpu, models_gpu, reference_outputs, queue);
      done_promise.set_value();
    }
    catch (...) {
      done_promise.set_exception(std::current_exception());
    }
  });

  ASSERT_EQ(done_future.wait_for(kLaunchTimeout), std::future_status::ready)
      << "launch_threads did not stop after gRPC startup failure";
  EXPECT_NO_THROW(done_future.get());
  EXPECT_TRUE(ctx.stop_requested.load(std::memory_order_relaxed));
  EXPECT_EQ(signal_stop_requested_flag(), 0);

  signal_stop_requested_flag() = 0;
}

TEST(ServerMainOrchestration, LaunchThreadsStopsOnBrutalSignal)
{
  if (running_under_tsan()) {
    GTEST_SKIP()
        << "Skipped under TSAN: gRPC event-engine triggers known external race "
           "in absl::raw_hash_set";
  }

  constexpr auto kLaunchTimeout = std::chrono::seconds(10);
  constexpr auto kServerStartTimeout = std::chrono::seconds(5);

  const int port = pick_unused_port();
  ASSERT_GT(port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(port);

  starpu_server::RuntimeConfig opts;
  opts.server_address = address;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 8;

  signal_stop_requested_flag() = 0;
  auto& ctx = server_context();
  ctx.stop_requested.store(false, std::memory_order_relaxed);

  starpu_server::StarPUSetup starpu(opts);
  torch::jit::script::Module model_cpu("m");
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> reference_outputs;
  starpu_server::InferenceQueue queue(opts.batching.max_queue_size);

  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();

  std::jthread launch_thread([&]() {
    try {
      launch_threads(
          opts, starpu, model_cpu, models_gpu, reference_outputs, queue);
      done_promise.set_value();
    }
    catch (...) {
      done_promise.set_exception(std::current_exception());
    }
  });

  ASSERT_TRUE(wait_for_channel_ready(address, kServerStartTimeout))
      << "Timed out waiting for gRPC server startup";

  std::latch storm_start{1};
  std::jthread signal_storm([&storm_start]() {
    storm_start.wait();
    for (int i = 0; i < 50; ++i) {
      signal_handler(SIGINT);
    }
  });
  storm_start.count_down();

  if (done_future.wait_for(kLaunchTimeout) != std::future_status::ready) {
    signal_stop_requested_flag() = 1;
    ctx.stop_requested.store(true, std::memory_order_relaxed);
    ctx.stop_cv.notify_one();
  }

  ASSERT_EQ(done_future.wait_for(kLaunchTimeout), std::future_status::ready)
      << "launch_threads did not stop within timeout on brutal signal";
  EXPECT_NO_THROW(done_future.get());

  signal_stop_requested_flag() = 0;
}

TEST(ServerMainOrchestration, LaunchThreadsStopsUnderConcurrentRpcLoad)
{
  if (running_under_tsan()) {
    GTEST_SKIP()
        << "Skipped under TSAN: gRPC event-engine triggers known external race "
           "in absl::raw_hash_set";
  }

  constexpr auto kLaunchTimeout = std::chrono::seconds(10);
  constexpr auto kClientActivityTimeout = std::chrono::seconds(2);
  constexpr int kClientThreads = 6;

  const int port = pick_unused_port();
  ASSERT_GT(port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(port);

  starpu_server::RuntimeConfig opts;
  opts.server_address = address;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 16;

  signal_stop_requested_flag() = 0;
  auto& ctx = server_context();
  ctx.stop_requested.store(false, std::memory_order_relaxed);

  starpu_server::StarPUSetup starpu(opts);
  torch::jit::script::Module model_cpu("m");
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> reference_outputs;
  starpu_server::InferenceQueue queue(opts.batching.max_queue_size);

  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();
  std::jthread launch_thread([&]() {
    try {
      launch_threads(
          opts, starpu, model_cpu, models_gpu, reference_outputs, queue);
      done_promise.set_value();
    }
    catch (...) {
      done_promise.set_exception(std::current_exception());
    }
  });

  auto channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  ASSERT_TRUE(channel->WaitForConnected(
      std::chrono::system_clock::now() + std::chrono::seconds(5)));

  std::atomic<bool> stop_clients{false};
  std::atomic<int> started_requests{0};
  std::mutex started_requests_mutex;
  std::condition_variable started_requests_cv;
  std::vector<std::jthread> clients;
  clients.reserve(kClientThreads);
  for (int i = 0; i < kClientThreads; ++i) {
    clients.emplace_back([&, channel]() {
      auto stub = inference::GRPCInferenceService::NewStub(channel);
      while (!stop_clients.load(std::memory_order_acquire)) {
        grpc::ClientContext rpc_ctx;
        rpc_ctx.set_deadline(
            std::chrono::system_clock::now() + std::chrono::milliseconds(200));
        inference::ServerLiveRequest request;
        inference::ServerLiveResponse response;
        const int previous =
            started_requests.fetch_add(1, std::memory_order_relaxed);
        if (previous == 0) {
          std::lock_guard<std::mutex> lock(started_requests_mutex);
          started_requests_cv.notify_one();
        }
        (void)stub->ServerLive(&rpc_ctx, request, &response);
      }
    });
  }

  {
    std::unique_lock<std::mutex> lock(started_requests_mutex);
    ASSERT_TRUE(started_requests_cv.wait_for(
        lock, kClientActivityTimeout,
        [&started_requests]() {
          return started_requests.load(std::memory_order_relaxed) > 0;
        }))
        << "No RPC request started before stop signal";
  }
  signal_handler(SIGTERM);
  stop_clients.store(true, std::memory_order_release);
  clients.clear();

  if (done_future.wait_for(kLaunchTimeout) != std::future_status::ready) {
    signal_stop_requested_flag() = 1;
    ctx.stop_requested.store(true, std::memory_order_relaxed);
    ctx.stop_cv.notify_one();
  }

  ASSERT_EQ(done_future.wait_for(kLaunchTimeout), std::future_status::ready)
      << "launch_threads did not stop within timeout under RPC load";
  EXPECT_NO_THROW(done_future.get());
  EXPECT_GT(started_requests.load(std::memory_order_relaxed), 0);

  signal_stop_requested_flag() = 0;
}

TEST(
    ServerMainOrchestration,
    LaunchThreadsStopsOnSignalStormUnderConcurrentRpcLoad)
{
  if (running_under_tsan()) {
    GTEST_SKIP()
        << "Skipped under TSAN: gRPC event-engine triggers known external race "
           "in absl::raw_hash_set";
  }

  constexpr auto kLaunchTimeout = std::chrono::seconds(10);
  constexpr auto kClientActivityTimeout = std::chrono::seconds(2);
  constexpr int kClientThreads = 8;

  const int port = pick_unused_port();
  ASSERT_GT(port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(port);

  starpu_server::RuntimeConfig opts;
  opts.server_address = address;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 16;

  signal_stop_requested_flag() = 0;
  auto& ctx = server_context();
  ctx.stop_requested.store(false, std::memory_order_relaxed);

  starpu_server::StarPUSetup starpu(opts);
  torch::jit::script::Module model_cpu("m");
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> reference_outputs;
  starpu_server::InferenceQueue queue(opts.batching.max_queue_size);

  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();
  std::jthread launch_thread([&]() {
    try {
      launch_threads(
          opts, starpu, model_cpu, models_gpu, reference_outputs, queue);
      done_promise.set_value();
    }
    catch (...) {
      done_promise.set_exception(std::current_exception());
    }
  });

  auto channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  ASSERT_TRUE(channel->WaitForConnected(
      std::chrono::system_clock::now() + std::chrono::seconds(5)));

  std::atomic<bool> stop_clients{false};
  std::atomic<int> started_requests{0};
  std::mutex started_requests_mutex;
  std::condition_variable started_requests_cv;
  std::vector<std::jthread> clients;
  clients.reserve(kClientThreads);
  for (int i = 0; i < kClientThreads; ++i) {
    clients.emplace_back([&, channel]() {
      auto stub = inference::GRPCInferenceService::NewStub(channel);
      while (!stop_clients.load(std::memory_order_acquire)) {
        grpc::ClientContext rpc_ctx;
        rpc_ctx.set_deadline(
            std::chrono::system_clock::now() + std::chrono::milliseconds(200));
        inference::ServerLiveRequest request;
        inference::ServerLiveResponse response;
        const int previous =
            started_requests.fetch_add(1, std::memory_order_relaxed);
        if (previous == 0) {
          std::lock_guard<std::mutex> lock(started_requests_mutex);
          started_requests_cv.notify_one();
        }
        (void)stub->ServerLive(&rpc_ctx, request, &response);
      }
    });
  }

  {
    std::unique_lock<std::mutex> lock(started_requests_mutex);
    ASSERT_TRUE(started_requests_cv.wait_for(
        lock, kClientActivityTimeout,
        [&started_requests]() {
          return started_requests.load(std::memory_order_relaxed) > 0;
        }))
        << "No RPC request started before signal storm";
  }

  std::latch storm_start{1};
  std::jthread signal_storm([&storm_start]() {
    storm_start.wait();
    for (int i = 0; i < 64; ++i) {
      signal_handler((i % 2 == 0) ? SIGINT : SIGTERM);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  storm_start.count_down();

  if (done_future.wait_for(kLaunchTimeout) != std::future_status::ready) {
    signal_stop_requested_flag() = 1;
    ctx.stop_requested.store(true, std::memory_order_relaxed);
    ctx.stop_cv.notify_one();
  }

  stop_clients.store(true, std::memory_order_release);
  clients.clear();

  ASSERT_EQ(done_future.wait_for(kLaunchTimeout), std::future_status::ready)
      << "launch_threads did not stop within timeout on signal storm under "
         "RPC load";
  EXPECT_NO_THROW(done_future.get());
  EXPECT_GT(started_requests.load(std::memory_order_relaxed), 0);

  signal_stop_requested_flag() = 0;
}

TEST(
    ServerMainPlotScript,
    ResolvePythonExecutableSkipsCandidatesThatAreNotRegularFiles)
{
  ResolvePythonExecutableOverrideState override_state;
  override_state.candidates = {
      "/tmp/python_candidate_not_regular_0",
      "/tmp/python_candidate_not_regular_1"};
  override_state.is_regular_file_result = false;
  override_state.force_status_error = false;
  ScopedResolvePythonExecutableOverrides overrides(override_state);

  const auto python_path = resolve_python_executable();
  EXPECT_FALSE(python_path.has_value());
  EXPECT_EQ(override_state.candidates_calls, 1);
  EXPECT_EQ(override_state.is_regular_file_calls, 2);
  EXPECT_EQ(override_state.observed_candidates, override_state.candidates);
}

TEST(
    ServerMainPlotScript,
    ResolvePythonExecutableReturnsNulloptWhenFileStatusReportsError)
{
  ResolvePythonExecutableOverrideState override_state;
  override_state.candidates = {"/tmp/python_candidate_status_error"};
  override_state.is_regular_file_result = true;
  override_state.force_status_error = true;
  ScopedResolvePythonExecutableOverrides overrides(override_state);

  const auto python_path = resolve_python_executable();
  EXPECT_FALSE(python_path.has_value());
  EXPECT_EQ(override_state.candidates_calls, 1);
  EXPECT_EQ(override_state.is_regular_file_calls, 1);
  ASSERT_EQ(override_state.observed_candidates.size(), 1U);
  EXPECT_EQ(
      override_state.observed_candidates[0],
      std::filesystem::path("/tmp/python_candidate_status_error"));
}

TEST(ServerMainPlotScript, LocatePlotScriptFindsRepositoryScript)
{
  starpu_server::RuntimeConfig opts;
  const auto script_path = locate_plot_script(opts);
  ASSERT_TRUE(script_path.has_value());
  EXPECT_TRUE(script_path->is_absolute());
  EXPECT_EQ(script_path->filename(), "plot_batch_summary.py");
  EXPECT_TRUE(std::filesystem::is_regular_file(*script_path));
}

TEST(ServerMainPlotPath, PlotsOutputPathRemovesSummarySuffixWhenPresent)
{
  const std::filesystem::path summary_path{"/tmp/batch_summary.csv"};
  const auto output_path = plots_output_path(summary_path);
  EXPECT_EQ(output_path, std::filesystem::path("/tmp/batch_plots.png"));
}

TEST(ServerMainPlotPath, PlotsOutputPathKeepsStemWhenSummarySuffixAbsent)
{
  const std::filesystem::path summary_path{"/tmp/trace.csv"};
  const auto output_path = plots_output_path(summary_path);
  EXPECT_EQ(output_path, std::filesystem::path("/tmp/trace_plots.png"));
}

TEST(ServerMainPlotPath, PlotsOutputPathUsesLastSummaryOccurrence)
{
  const std::filesystem::path summary_path{"/tmp/a_summary_v2_summary.csv"};
  const auto output_path = plots_output_path(summary_path);
  EXPECT_EQ(output_path, std::filesystem::path("/tmp/a_summary_v2_plots.png"));
}

TEST(ServerMainPlotPath, RunTracePlotsReturnsImmediatelyWhenTracingDisabled)
{
  ScopedTraceLoggerReset trace_logger_reset;

  PlotScriptOverrideState override_state;
  override_state.summary_path_result = std::filesystem::path("/tmp/unused.csv");
  override_state.locate_result = std::filesystem::path("/tmp/fake_plot.py");
  override_state.run_result = 0;
  ScopedPlotScriptOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.batching.trace_enabled = false;

  OStreamCapture capture_err(std::cerr);
  OStreamCapture capture_out(std::cout);
  run_trace_plots_if_enabled(opts);

  EXPECT_EQ(override_state.summary_path_calls, 0);
  EXPECT_EQ(override_state.locate_calls, 0);
  EXPECT_EQ(override_state.run_calls, 0);
  EXPECT_EQ(capture_err.str(), "");
  EXPECT_EQ(capture_out.str(), "");
}

TEST(ServerMainPlotPath, RunTracePlotsWarnsWhenNoSummaryFileIsAvailable)
{
  ScopedTraceLoggerReset trace_logger_reset;

  PlotScriptOverrideState override_state;
  override_state.summary_path_result = std::nullopt;
  override_state.locate_result = std::filesystem::path("/tmp/fake_plot.py");
  override_state.run_result = 0;
  ScopedPlotScriptOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.batching.trace_enabled = true;

  OStreamCapture capture_err(std::cerr);
  OStreamCapture capture_out(std::cout);
  run_trace_plots_if_enabled(opts);

  EXPECT_EQ(override_state.summary_path_calls, 1);
  EXPECT_EQ(override_state.locate_calls, 0);
  EXPECT_EQ(override_state.run_calls, 0);
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log(
          "Tracing was enabled but no trace.csv was produced; skipping plot "
          "generation."));
  EXPECT_EQ(capture_out.str(), "");
}

TEST(ServerMainPlotPath, RunTracePlotsWarnsWhenSummaryFileIsMissing)
{
  ScopedTraceLoggerReset trace_logger_reset;
  auto temp_directory = make_temp_test_directory("run_trace_plots_missing_csv");
  const auto summary_path = temp_directory.path / "trace_summary.csv";
  ASSERT_FALSE(std::filesystem::exists(summary_path));

  PlotScriptOverrideState override_state;
  override_state.summary_path_result = summary_path;
  override_state.locate_result = std::filesystem::path("/tmp/fake_plot.py");
  override_state.run_result = 0;
  ScopedPlotScriptOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.batching.trace_enabled = true;

  OStreamCapture capture_err(std::cerr);
  OStreamCapture capture_out(std::cout);
  run_trace_plots_if_enabled(opts);

  EXPECT_EQ(override_state.summary_path_calls, 1);
  EXPECT_EQ(override_state.locate_calls, 0);
  EXPECT_EQ(override_state.run_calls, 0);
  EXPECT_EQ(
      capture_err.str(), expected_warning_log(
                             "Tracing summary file '" + summary_path.string() +
                             "' not found; skipping plot generation."));
  EXPECT_EQ(capture_out.str(), "");
}

TEST(ServerMainPlotPath, RunTracePlotsWarnsWhenPlotScriptCannotBeLocated)
{
  ScopedTraceLoggerReset trace_logger_reset;
  auto temp_directory =
      make_temp_test_directory("run_trace_plots_missing_script");
  const auto summary_path = temp_directory.path / "trace_summary.csv";
  TempFileGuard summary_guard{summary_path};
  {
    std::ofstream summary_file(summary_path);
    ASSERT_TRUE(summary_file.is_open());
    summary_file << "batch_id\n";
  }
  ASSERT_TRUE(std::filesystem::exists(summary_path));

  PlotScriptOverrideState override_state;
  override_state.summary_path_result = summary_path;
  override_state.locate_result = std::nullopt;
  override_state.run_result = 0;
  ScopedPlotScriptOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.batching.trace_enabled = true;

  OStreamCapture capture_err(std::cerr);
  OStreamCapture capture_out(std::cout);
  run_trace_plots_if_enabled(opts);

  EXPECT_EQ(override_state.summary_path_calls, 1);
  EXPECT_EQ(override_state.locate_calls, 1);
  EXPECT_EQ(override_state.run_calls, 0);
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log(
          "Unable to locate scripts/plot_batch_summary.py; skipping plot "
          "generation."));
  EXPECT_EQ(capture_out.str(), "");
}

TEST(ServerMainPlotPath, RunTracePlotsWarnsWhenPlotScriptDidNotComplete)
{
  ScopedTraceLoggerReset trace_logger_reset;
  auto temp_directory =
      make_temp_test_directory("run_trace_plots_script_incomplete");
  const auto summary_path = temp_directory.path / "trace_summary.csv";
  TempFileGuard summary_guard{summary_path};
  {
    std::ofstream summary_file(summary_path);
    ASSERT_TRUE(summary_file.is_open());
    summary_file << "batch_id\n";
  }
  ASSERT_TRUE(std::filesystem::exists(summary_path));

  PlotScriptOverrideState override_state;
  override_state.summary_path_result = summary_path;
  override_state.locate_result = temp_directory.path / "plot_batch_summary.py";
  override_state.run_result = std::nullopt;
  ScopedPlotScriptOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.batching.trace_enabled = true;

  OStreamCapture capture_err(std::cerr);
  OStreamCapture capture_out(std::cout);
  run_trace_plots_if_enabled(opts);

  ASSERT_TRUE(override_state.locate_result.has_value());
  EXPECT_EQ(override_state.summary_path_calls, 1);
  EXPECT_EQ(override_state.locate_calls, 1);
  EXPECT_EQ(override_state.run_calls, 1);
  EXPECT_EQ(override_state.run_script_path, *override_state.locate_result);
  EXPECT_EQ(override_state.run_summary_path, summary_path);
  EXPECT_EQ(override_state.run_output_path, plots_output_path(summary_path));
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log(
          "Failed to generate batching latency plots; plot script did not "
          "complete."));
  EXPECT_EQ(capture_out.str(), "");
}

TEST(ServerMainPlotPath, RunTracePlotsWarnsWhenPlotScriptFails)
{
  ScopedTraceLoggerReset trace_logger_reset;
  auto temp_directory =
      make_temp_test_directory("run_trace_plots_script_failed");
  const auto summary_path = temp_directory.path / "trace_summary.csv";
  TempFileGuard summary_guard{summary_path};
  {
    std::ofstream summary_file(summary_path);
    ASSERT_TRUE(summary_file.is_open());
    summary_file << "batch_id\n";
  }
  ASSERT_TRUE(std::filesystem::exists(summary_path));

  PlotScriptOverrideState override_state;
  override_state.summary_path_result = summary_path;
  override_state.locate_result = temp_directory.path / "plot_batch_summary.py";
  override_state.run_result = 42;
  ScopedPlotScriptOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.batching.trace_enabled = true;

  const auto expected_output_path = plots_output_path(summary_path);
  const auto expected_message = std::format(
      "Failed to generate batching latency plots; python3 {} {} --output {} "
      "exited with code {}.",
      override_state.locate_result->string(), summary_path.string(),
      expected_output_path.string(), *override_state.run_result);

  OStreamCapture capture_err(std::cerr);
  OStreamCapture capture_out(std::cout);
  run_trace_plots_if_enabled(opts);

  EXPECT_EQ(override_state.summary_path_calls, 1);
  EXPECT_EQ(override_state.locate_calls, 1);
  EXPECT_EQ(override_state.run_calls, 1);
  EXPECT_EQ(override_state.run_output_path, expected_output_path);
  EXPECT_EQ(capture_err.str(), expected_warning_log(expected_message));
  EXPECT_EQ(capture_out.str(), "");
}

TEST(ServerMainPlotPath, RunTracePlotsLogsInfoWhenPlotScriptSucceeds)
{
  ScopedTraceLoggerReset trace_logger_reset;
  auto temp_directory = make_temp_test_directory("run_trace_plots_script_ok");
  const auto summary_path = temp_directory.path / "trace_summary.csv";
  TempFileGuard summary_guard{summary_path};
  {
    std::ofstream summary_file(summary_path);
    ASSERT_TRUE(summary_file.is_open());
    summary_file << "batch_id\n";
  }
  ASSERT_TRUE(std::filesystem::exists(summary_path));

  PlotScriptOverrideState override_state;
  override_state.summary_path_result = summary_path;
  override_state.locate_result = temp_directory.path / "plot_batch_summary.py";
  override_state.run_result = 0;
  ScopedPlotScriptOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.batching.trace_enabled = true;
  opts.verbosity = starpu_server::VerbosityLevel::Info;

  const auto expected_output_path = plots_output_path(summary_path);
  const auto expected_message = std::format(
      "Batching latency plots written to '{}'.", expected_output_path.string());

  OStreamCapture capture_err(std::cerr);
  OStreamCapture capture_out(std::cout);
  run_trace_plots_if_enabled(opts);

  EXPECT_EQ(override_state.summary_path_calls, 1);
  EXPECT_EQ(override_state.locate_calls, 1);
  EXPECT_EQ(override_state.run_calls, 1);
  EXPECT_EQ(override_state.run_output_path, expected_output_path);
  EXPECT_EQ(capture_err.str(), "");
  EXPECT_EQ(capture_out.str(), expected_info_log(expected_message));
}

TEST(ServerMainPlotProcess, WaitForPlotProcessReturnsChildExitCode)
{
  const pid_t pid = spawn_exiting_child(7);
  ASSERT_GT(pid, 0);
  const auto exit_code = wait_for_plot_process(pid);
  ASSERT_TRUE(exit_code.has_value());
  EXPECT_EQ(*exit_code, 7);
}

TEST(ServerMainPlotProcess, LogWaitpidErrorLogsWarningWithErrnoMessage)
{
  const int saved_errno = errno;
  errno = ECHILD;

  OStreamCapture capture_err(std::cerr);
  log_waitpid_error();

  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log(
          std::string("Failed to wait for plot generation process: ") +
          std::strerror(ECHILD)));

  errno = saved_errno;
}

TEST(ServerMainPlotProcess, WaitForExitWithTimeoutTimesOutThenReapsChild)
{
  const pid_t pid = spawn_sleeping_child(false);
  ASSERT_GT(pid, 0);

  const auto result =
      wait_for_exit_with_timeout(pid, std::chrono::milliseconds(10));
  EXPECT_EQ(result.outcome, WaitOutcome::TimedOut);
  EXPECT_FALSE(result.exit_code.has_value());

  ASSERT_EQ(::kill(pid, SIGKILL), 0);
  const auto exit_code = wait_for_exit_blocking(pid);
  ASSERT_TRUE(exit_code.has_value());
  EXPECT_EQ(*exit_code, 128 + SIGKILL);
}

TEST(
    ServerMainPlotProcess, TerminateAndWaitEscalatesToSigkillWhenSigtermIgnored)
{
  const pid_t pid = spawn_sleeping_child(true);
  ASSERT_GT(pid, 0);

  const auto exit_code = terminate_and_wait(pid);
  ASSERT_TRUE(exit_code.has_value());
  EXPECT_EQ(*exit_code, 128 + SIGKILL);
}

TEST(ServerMainPlotProcess, RunPlotScriptReturnsNonZeroForMissingScript)
{
  const auto summary_path = make_temp_test_path("batch_summary", ".csv");
  const auto output_path = make_temp_test_path("batch_plots", ".png");
  TempFileGuard summary_guard{summary_path};
  TempFileGuard output_guard{output_path};
  {
    std::ofstream summary_file(summary_path);
    ASSERT_TRUE(summary_file.is_open());
    summary_file << "timestamp_ms,latency_ms\n";
  }

  const auto exit_code = run_plot_script(
      "/definitely/missing_plot_batch_summary.py", summary_path, output_path);
  if (!resolve_python_executable().has_value()) {
    EXPECT_FALSE(exit_code.has_value());
    return;
  }

  ASSERT_TRUE(exit_code.has_value());
  EXPECT_NE(*exit_code, 0);
}

}  // namespace
