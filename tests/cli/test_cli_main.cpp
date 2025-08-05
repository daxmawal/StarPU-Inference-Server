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
  EXPECT_NE(output.find(kCliHelpMessage), std::string::npos);
}

class CliMainInvalidOptionsDeath
    : public ::testing::TestWithParam<std::vector<const char*>> {};

TEST_P(CliMainInvalidOptionsDeath, Fatal)
{
  const auto& args = GetParam();
  auto argv = build_argv(args);
  EXPECT_DEATH(
      { cli_main(static_cast<int>(argv.size()), argv.data()); },
      kInvalidOptionsRegex);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidArguments, CliMainInvalidOptionsDeath,
    ::testing::Values(
        std::vector<const char*>{
            "program", "--model", "model.pt", "--shapes", "1x2,2x3", "--types",
            "float"},
        std::vector<const char*>{
            "program", "--shape", "1x1", "--types", "float"},
        std::vector<const char*>{
            "program", "--model", "model.pt", "--shape", "1x1"}));

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
