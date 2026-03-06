#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
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
  const auto suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
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
  constexpr auto kSignalDelay = std::chrono::milliseconds(100);

  starpu_server::RuntimeConfig opts;
  opts.server_address = "127.0.0.1:0";
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

  std::this_thread::sleep_for(kSignalDelay);
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

}  // namespace
