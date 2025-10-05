#include <gtest/gtest.h>

#include <functional>
#include <optional>
#include <string>

#include "test_cli.hpp"
#include "utils/config_loader.hpp"
#include "utils/perf_observer.hpp"
#include "utils/runtime_config.hpp"

namespace starpu_server {
struct RuntimeConfig;
class StarPUSetup;
using RunLoopPtr = void (*)(const RuntimeConfig&, StarPUSetup&);
static RunLoopPtr run_inference_loop_hook = nullptr;

using LoadConfigHook = std::function<RuntimeConfig(const std::string&)>;
static LoadConfigHook load_config_hook;
static auto (*const load_config_default)(const std::string&)
    -> RuntimeConfig = ::starpu_server::load_config;

struct RunLoopHookGuard {
  explicit RunLoopHookGuard(RunLoopPtr hook) { run_inference_loop_hook = hook; }
  ~RunLoopHookGuard() { run_inference_loop_hook = nullptr; }
  RunLoopHookGuard(const RunLoopHookGuard&) = delete;
  auto operator=(const RunLoopHookGuard&) -> RunLoopHookGuard& = delete;
  RunLoopHookGuard(RunLoopHookGuard&&) = delete;
  auto operator=(RunLoopHookGuard&&) -> RunLoopHookGuard& = delete;
};

inline void
fake_run_inference_loop(const RuntimeConfig& opts, StarPUSetup& starpu)
{
  run_inference_loop_hook(opts, starpu);
}

inline auto
fake_load_config(const std::string& path) -> RuntimeConfig
{
  if (load_config_hook) {
    return load_config_hook(path);
  }
  return load_config_default(path);
}

namespace perf_observer {
using SnapshotHook = std::function<std::optional<Snapshot>()>;
static SnapshotHook snapshot_hook;
static auto (*const snapshot_default)()
    -> std::optional<Snapshot> = ::starpu_server::perf_observer::snapshot;

inline auto
fake_snapshot() -> std::optional<Snapshot>
{
  if (snapshot_hook) {
    return snapshot_hook();
  }
  return snapshot_default();
}
}  // namespace perf_observer
}  // namespace starpu_server
#define run_inference_loop fake_run_inference_loop
#define load_config fake_load_config
#define snapshot fake_snapshot
#define main cli_main
#include "cli/main.cpp"
#undef main
#undef snapshot
#undef load_config
#undef run_inference_loop

namespace {
starpu_server::RuntimeConfig g_runloop_config;
bool g_runloop_called = false;
bool* g_runloop_flag = nullptr;

void
capture_runtime_config(
    const starpu_server::RuntimeConfig& opts, starpu_server::StarPUSetup&)
{
  g_runloop_config = opts;
  g_runloop_called = true;
  if (g_runloop_flag != nullptr) {
    *g_runloop_flag = true;
  }
}
}  // namespace

class CliMain_Integration : public ::testing::Test {
 protected:
  void SetUp() override
  {
    g_runloop_config = {};
    g_runloop_called = false;
    g_runloop_flag = nullptr;
  }

  void TearDown() override
  {
    starpu_server::run_inference_loop_hook = nullptr;
    starpu_server::load_config_hook = nullptr;
    starpu_server::perf_observer::snapshot_hook = nullptr;
    g_runloop_flag = nullptr;
  }
};

TEST_F(CliMain_Integration, ShowsHelpMessage)
{
  auto argv = build_argv({"program", "--help"});
  testing::internal::CaptureStdout();
  int return_code = cli_main(static_cast<int>(argv.size()), argv.data());
  std::string out = testing::internal::GetCapturedStdout();
  EXPECT_EQ(return_code, 0);
  const auto help = starpu_server::get_help_message("Inference Engine");
  EXPECT_NE(out.find(help), std::string::npos);
}

namespace starpu_server {
[[noreturn]] static void
throw_inference_error(const RuntimeConfig& /*unused*/, StarPUSetup& /*unused*/)
{
  throw InferenceEngineException("fail");
}
[[noreturn]] static void
throw_std_error(const RuntimeConfig& /*unused*/, StarPUSetup& /*unused*/)
{
  throw std::runtime_error("boom");
}
}  // namespace starpu_server

TEST_F(CliMain_Integration, ReturnsTwoOnInferenceEngineException)
{
  starpu_server::RunLoopHookGuard guard(starpu_server::throw_inference_error);
  auto argv = build_valid_cli_args();
  int return_code = cli_main(static_cast<int>(argv.size()), argv.data());
  EXPECT_EQ(return_code, 2);
}

TEST_F(CliMain_Integration, ReturnsMinusOneOnStdException)
{
  starpu_server::RunLoopHookGuard guard(starpu_server::throw_std_error);
  auto argv = build_valid_cli_args();
  int return_code = cli_main(static_cast<int>(argv.size()), argv.data());
  EXPECT_EQ(return_code, -1);
}

TEST_F(CliMain_Integration, UsesConfigFileWhenProvided)
{
  const std::string& config_path = test_config_path();
  bool hook_invoked = false;
  std::string received_path;
  starpu_server::load_config_hook =
      [&](const std::string& path) -> starpu_server::RuntimeConfig {
    hook_invoked = true;
    received_path = path;
    starpu_server::RuntimeConfig cfg;
    cfg.scheduler = "from-config";
    return cfg;
  };

  bool run_loop_executed = false;
  g_runloop_flag = &run_loop_executed;
  starpu_server::RunLoopHookGuard guard(capture_runtime_config);
  auto argv = build_argv(
      {"program", "--config", config_path.c_str(), "--model",
       test_model_path().c_str(), "--shape", "1x1", "--types", "float"});

  const int return_code = cli_main(static_cast<int>(argv.size()), argv.data());

  EXPECT_EQ(return_code, 0);
  EXPECT_TRUE(hook_invoked);
  EXPECT_EQ(received_path, config_path);
  EXPECT_TRUE(run_loop_executed);
  EXPECT_TRUE(g_runloop_called);
  EXPECT_EQ(g_runloop_config.config_path, config_path);
  EXPECT_EQ(g_runloop_config.scheduler, "from-config");
}

TEST_F(CliMain_Integration, ExitsOnInvalidOptions)
{
  EXPECT_DEATH(
      {
        auto argv = build_argv({"program", "--model", "missing"});
        (void)cli_main(static_cast<int>(argv.size()), argv.data());
      },
      "Invalid program options");
}

TEST_F(CliMain_Integration, LogsStatsWhenSnapshotAvailable)
{
  starpu_server::perf_observer::snapshot_hook = []() {
    return starpu_server::perf_observer::Snapshot{
        .total_inferences = 4,
        .duration_seconds = 2.0,
        .throughput = 2.0,
    };
  };

  starpu_server::RunLoopHookGuard guard(capture_runtime_config);
  auto argv = build_argv(
      {"program", "--model", test_model_path().c_str(), "--shape", "1x1",
       "--types", "float", "--verbose", "stats"});

  testing::internal::CaptureStdout();
  const int return_code = cli_main(static_cast<int>(argv.size()), argv.data());
  std::string out = testing::internal::GetCapturedStdout();

  EXPECT_EQ(return_code, 0);
  EXPECT_TRUE(g_runloop_called);
  EXPECT_NE(
      out.find("Throughput: 2.000 inf/s (4 inferences over 2.000 s)"),
      std::string::npos);
}

TEST_F(CliMain_Integration, ReturnsZeroOnSuccess)
{
  bool run_loop_executed = false;
  g_runloop_flag = &run_loop_executed;
  starpu_server::RunLoopHookGuard guard(capture_runtime_config);
  starpu_server::perf_observer::snapshot_hook = []() { return std::nullopt; };

  auto argv = build_valid_cli_args();
  const int return_code = cli_main(static_cast<int>(argv.size()), argv.data());

  EXPECT_EQ(return_code, 0);
  EXPECT_TRUE(run_loop_executed);
  EXPECT_TRUE(g_runloop_called);
}
