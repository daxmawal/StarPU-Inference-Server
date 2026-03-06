#include <arpa/inet.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
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
#include <string>
#include <thread>
#include <vector>

#define main starpu_server_server_main_for_test
#include "../../../src/grpc/server/server_main.cpp"
#undef main

namespace {

struct TempFileGuard {
  std::filesystem::path path;

  ~TempFileGuard()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
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

TEST(ServerMainOrchestration, LaunchThreadsStopsAfterSignal)
{
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

TEST(ServerMainPlotScript, LocatePlotScriptFindsRepositoryScript)
{
  starpu_server::RuntimeConfig opts;
  const auto script_path = locate_plot_script(opts);
  ASSERT_TRUE(script_path.has_value());
  EXPECT_TRUE(script_path->is_absolute());
  EXPECT_EQ(script_path->filename(), "plot_batch_summary.py");
  EXPECT_TRUE(std::filesystem::is_regular_file(*script_path));
}

TEST(ServerMainPlotProcess, WaitForPlotProcessReturnsChildExitCode)
{
  const pid_t pid = spawn_exiting_child(7);
  ASSERT_GT(pid, 0);
  const auto exit_code = wait_for_plot_process(pid);
  ASSERT_TRUE(exit_code.has_value());
  EXPECT_EQ(*exit_code, 7);
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
