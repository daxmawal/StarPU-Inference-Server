#include <gtest/gtest.h>

#include <vector>

#include "cli/args_parser.hpp"
#include "cli_test_utils.hpp"
#include "utils/exceptions.hpp"
#include "utils/runtime_config.hpp"

static const std::vector<const char*> kCommonArgs = {
    "program", "--model", "model.pt", "--shape", "1x3", "--types", "float"};

TEST(ArgsParser, ParsesRequiredOptions)
{
  const auto opts = parse(
      {"program", "--model", "model.pt", "--shape", "1x3x224x224", "--types",
       "float"});
  ASSERT_TRUE(opts.valid);
  EXPECT_EQ(opts.model_path, "model.pt");
  ASSERT_EQ(opts.input_shapes.size(), 1U);
  EXPECT_EQ(opts.input_shapes[0], std::vector<int64_t>({1, 3, 224, 224}));
  ASSERT_EQ(opts.input_types.size(), 1U);
  EXPECT_EQ(opts.input_types[0], at::kFloat);
}

TEST(ArgsParser, ParsesAllOptions)
{
  const auto opts = parse(
      {"program",
       "--model",
       "model.pt",
       "--shapes",
       "1x3x224x224,2x1",
       "--types",
       "float,int",
       "--iterations",
       "5",
       "--device-ids",
       "0,1",
       "--verbose",
       "3",
       "--delay",
       "42",
       "--scheduler",
       "lws",
       "--address",
       "127.0.0.1:1234",
       "--max-msg-size",
       "512",
       "--sync",
       "--no_cpu"});
  ASSERT_TRUE(opts.valid);
  EXPECT_EQ(opts.scheduler, "lws");
  EXPECT_EQ(opts.model_path, "model.pt");
  EXPECT_EQ(opts.iterations, 5);
  EXPECT_EQ(opts.delay_ms, 42);
  EXPECT_EQ(opts.verbosity, starpu_server::VerbosityLevel::Debug);
  EXPECT_EQ(opts.server_address, "127.0.0.1:1234");
  EXPECT_EQ(opts.max_message_bytes, 512);
  EXPECT_TRUE(opts.synchronous);
  EXPECT_FALSE(opts.use_cpu);
  EXPECT_TRUE(opts.use_cuda);
  ASSERT_EQ(opts.device_ids, (std::vector<int>{0, 1}));
  ASSERT_EQ(opts.input_shapes.size(), 2U);
  EXPECT_EQ(opts.input_shapes[0], std::vector<int64_t>({1, 3, 224, 224}));
  EXPECT_EQ(opts.input_shapes[1], std::vector<int64_t>({2, 1}));
  ASSERT_EQ(opts.input_types.size(), 2U);
  EXPECT_EQ(opts.input_types[0], at::kFloat);
  EXPECT_EQ(opts.input_types[1], at::kInt);
}

class ArgsParserInvalidOptions
    : public ::testing::TestWithParam<std::vector<const char*>> {};

TEST_P(ArgsParserInvalidOptions, Invalid)
{
  auto args = kCommonArgs;
  const auto& diff = GetParam();
  args.insert(args.end(), diff.begin(), diff.end());
  expect_invalid(args);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidArguments, ArgsParserInvalidOptions,
    ::testing::Values(
        std::vector<const char*>{"--types", ""},
        std::vector<const char*>{"--iterations", "0"},
        std::vector<const char*>{"--shape", "1x-3x3"},
        std::vector<const char*>{"--types", "unknown"},
        std::vector<const char*>{"--device-ids", "-1"},
        std::vector<const char*>{"--unknown"},
        std::vector<const char*>{"--shapes", "1x2,2x3", "--types", "float"},
        std::vector<const char*>{"--shape", "1xax2"},
        std::vector<const char*>{"--shape", "9223372036854775808"},
        std::vector<const char*>{"--verbose", "5"},
        std::vector<const char*>{"--shape", ""},
        std::vector<const char*>{"--shapes", ""},
        std::vector<const char*>{"--shapes", "1x2,", "--types", "float"},
        std::vector<const char*>{"--iterations", "-1"},
        std::vector<const char*>{"--delay", "-1"},
        std::vector<const char*>{"--max-msg-size", "0"},
        std::vector<const char*>{"--max-msg-size", "-1"},
        std::vector<const char*>{"--shapes", "1x2,,3", "--types", "float,int"},
        std::vector<const char*>{"--model"},
        std::vector<const char*>{"--iterations"}));

TEST(ArgsParser, VerboseLevels)
{
  using enum starpu_server::VerbosityLevel;
  const std::array<std::pair<const char*, starpu_server::VerbosityLevel>, 5>
      cases = {{
          {"0", Silent},
          {"1", Info},
          {"2", Stats},
          {"3", Debug},
          {"4", Trace},
      }};
  for (const auto& [level_str, expected] : cases) {
    const auto opts = parse(
        {"program", "--model", "model.pt", "--shape", "1x3", "--types", "float",
         "--verbose", level_str});
    ASSERT_TRUE(opts.valid);
    EXPECT_EQ(opts.verbosity, expected);
  }
}

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
  const auto help_message = starpu_server::get_help_message("Inference Engine");
  EXPECT_NE(output.find(help_message), std::string::npos);
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
[[noreturn]] static void
throw_inference_error(const RuntimeConfig& /*config*/, StarPUSetup& /*setup*/)
{
  throw InferenceEngineException("fail");
}

[[noreturn]] static void
throw_std_error(const RuntimeConfig& /*config*/, StarPUSetup& /*setup*/)
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
