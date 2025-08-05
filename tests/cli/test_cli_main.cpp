#include <gtest/gtest.h>

#include "cli/args_parser.hpp"
#include "cli_test_utils.hpp"
#include "utils/exceptions.hpp"

constexpr char kInvalidOptionsRegex[] = "Invalid program options\\.";

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

TEST(CliMain, ShowsHelpMessage)
{
  auto argv = build_argv({"program", "--help"});
  testing::internal::CaptureStdout();
  int result = cli_main(static_cast<int>(argv.size()), argv.data());
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(result, 0);
  const std::string expected =
      "Usage: Inference Engine [OPTIONS]\n"
      "\nOptions:\n"
      "  --scheduler [name]      Scheduler type (default: lws)\n"
      "  --model [path]          Path to TorchScript model file (.pt)\n"
      "  --iterations [num]      Number of iterations (default: 1)\n"
      "  --shape 1x3x224x224     Shape of a single input tensor\n"
      "  --shapes shape1,shape2  Shapes for multiple input tensors\n"
      "  --types float,int       Input tensor types (default: float)\n"
      "  --sync                  Run tasks in synchronous mode\n"
      "  --delay [ms]            Delay between jobs (default: 0)\n"
      "  --no_cpu                Disable CPU usage\n"
      "  --device-ids 0,1        GPU device IDs for inference\n"
      "  --address ADDR          gRPC server listen address\n"
      "  --max-msg-size BYTES    Max gRPC message size in bytes\n"
      "  --verbose [0-4]         Verbosity level: 0=silent to 4=trace\n"
      "  --help                  Show this help message\n";
  EXPECT_EQ(output, expected);
}

TEST(CliMain, InvalidOptionsDeath)
{
  auto argv = build_argv(
      {"program", "--model", "model.pt", "--shapes", "1x2,2x3", "--types",
       "float"});
  EXPECT_DEATH(
      { cli_main(static_cast<int>(argv.size()), argv.data()); },
      kInvalidOptionsRegex);
}

namespace starpu_server {
static void
throw_inference_error(const RuntimeConfig&, StarPUSetup&)
{
  throw InferenceEngineException("fail");
}

static void
throw_std_error(const RuntimeConfig&, StarPUSetup&)
{
  throw std::runtime_error("boom");
}
}  // namespace starpu_server

TEST(CliMain, ReturnsTwoOnInferenceEngineException)
{
  starpu_server::RunLoopHookGuard guard(starpu_server::throw_inference_error);
  auto argv = build_valid_cli_args();
  int result = cli_main(static_cast<int>(argv.size()), argv.data());
  EXPECT_EQ(result, 2);
}

TEST(CliMain, ReturnsMinusOneOnStdException)
{
  starpu_server::RunLoopHookGuard guard(starpu_server::throw_std_error);
  auto argv = build_valid_cli_args();
  int result = cli_main(static_cast<int>(argv.size()), argv.data());
  EXPECT_EQ(result, -1);
}

TEST(CliMain, ReturnsFatalOnInvalidOptions)
{
  auto argv = build_argv(
      {"program", "--model", "model.pt", "--shapes", "1x2,2x3", "--types",
       "float"});
  EXPECT_DEATH(
      { cli_main(static_cast<int>(argv.size()), argv.data()); },
      kInvalidOptionsRegex);
}
