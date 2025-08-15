#include <gtest/gtest.h>

#include "test_cli.hpp"
#include "utils/runtime_config.hpp"

namespace starpu_server {
struct RuntimeConfig;
class StarPUSetup;
using RunLoopPtr = void (*)(const RuntimeConfig&, StarPUSetup&);
static RunLoopPtr run_inference_loop_hook = nullptr;

struct RunLoopHookGuard {
  explicit RunLoopHookGuard(RunLoopPtr hook) { run_inference_loop_hook = hook; }
  ~RunLoopHookGuard() { run_inference_loop_hook = nullptr; }
};

inline void
fake_run_inference_loop(const RuntimeConfig& opts, StarPUSetup& starpu)
{
  run_inference_loop_hook(opts, starpu);
}
}  // namespace starpu_server
#define run_inference_loop fake_run_inference_loop
#define main cli_main
#include "cli/main.cpp"
#undef main
#undef run_inference_loop

TEST(CliMain_Integration, ShowsHelpMessage)
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

TEST(CliMain_Integration, ReturnsTwoOnInferenceEngineException)
{
  starpu_server::RunLoopHookGuard guard(starpu_server::throw_inference_error);
  auto argv = build_valid_cli_args();
  int return_code = cli_main(static_cast<int>(argv.size()), argv.data());
  EXPECT_EQ(return_code, 2);
}

TEST(CliMain_Integration, ReturnsMinusOneOnStdException)
{
  starpu_server::RunLoopHookGuard guard(starpu_server::throw_std_error);
  auto argv = build_valid_cli_args();
  int return_code = cli_main(static_cast<int>(argv.size()), argv.data());
  EXPECT_EQ(return_code, -1);
}
