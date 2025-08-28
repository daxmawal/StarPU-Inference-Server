#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "test_cli.hpp"
#include "utils/runtime_config.hpp"

TEST(ArgsParser_Unit, ParsesRequiredOptions)
{
  const auto opts = parse(
      {"program", "--model", "model.pt", "--shape", "1x3x224x224", "--types",
       "float"});
  ASSERT_TRUE(opts.valid);
  EXPECT_EQ(opts.model_path, "model.pt");
  ASSERT_EQ(opts.inputs.size(), 1U);
  EXPECT_EQ(opts.inputs[0].dims, (std::vector<int64_t>{1, 3, 224, 224}));
  EXPECT_EQ(opts.inputs[0].type, at::kFloat);
}

TEST(ArgsParser_Unit, ParsesAllOptions)
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
       "--max-batch-size",
       "2",
       "--pregen-inputs",
       "7",
       "--warmup-iterations",
       "3",
       "--seed",
       "123",
       "--sync",
       "--no_cpu"});
  ASSERT_TRUE(opts.valid);
  EXPECT_EQ(opts.scheduler, "lws");
  EXPECT_EQ(opts.model_path, "model.pt");
  EXPECT_EQ(opts.iterations, 5);
  EXPECT_EQ(opts.delay_ms, 42);
  EXPECT_EQ(opts.verbosity, starpu_server::VerbosityLevel::Debug);
  EXPECT_EQ(opts.server_address, "127.0.0.1:1234");
  EXPECT_EQ(opts.max_batch_size, 2);
  EXPECT_EQ(opts.pregen_inputs, 7U);
  EXPECT_EQ(opts.warmup_iterations, 3);
  ASSERT_TRUE(opts.seed.has_value());
  EXPECT_EQ(opts.seed.value(), 123U);
  constexpr std::size_t expected_bytes = 32 * 1024 * 1024;
  EXPECT_EQ(opts.max_message_bytes, expected_bytes);
  EXPECT_TRUE(opts.synchronous);
  EXPECT_FALSE(opts.use_cpu);
  EXPECT_TRUE(opts.use_cuda);
  ASSERT_EQ(opts.device_ids, (std::vector<int>{0, 1}));
  ASSERT_EQ(opts.inputs.size(), 2U);
  EXPECT_EQ(opts.inputs[0].dims, (std::vector<int64_t>{1, 3, 224, 224}));
  EXPECT_EQ(opts.inputs[1].dims, (std::vector<int64_t>{2, 1}));
  EXPECT_EQ(opts.inputs[0].type, at::kFloat);
  EXPECT_EQ(opts.inputs[1].type, at::kInt);
}

TEST(ArgsParser_Unit, VerboseLevels)
{
  using enum starpu_server::VerbosityLevel;
  const std::array<std::pair<const char*, starpu_server::VerbosityLevel>, 5>
      cases = {
          {{"0", Silent},
           {"1", Info},
           {"2", Stats},
           {"3", Debug},
           {"4", Trace}}};
  for (const auto& [lvl, expected] : cases) {
    const auto opts = parse(
        {"program", "--model", "model.pt", "--shape", "1x3", "--types", "float",
         "--verbose", lvl});
    ASSERT_TRUE(opts.valid);
    EXPECT_EQ(opts.verbosity, expected);
  }
}

TEST(ArgsParser_Unit, ParsesMixedCaseTypes)
{
  const auto opts = parse(
      {"program", "--model", "model.pt", "--shapes", "1x1,1x1", "--types",
       "FlOat,InT"});
  ASSERT_TRUE(opts.valid);
  ASSERT_EQ(opts.inputs.size(), 2U);
  EXPECT_EQ(opts.inputs[0].type, at::kFloat);
  EXPECT_EQ(opts.inputs[1].type, at::kInt);
}

TEST(ArgsParser_Unit, MetricsPortBoundaryValues)
{
  for (const int port : {1, 65535}) {
    std::string port_str = std::to_string(port);
    std::vector<const char*> args = {
        "program", "--model", "model.pt",       "--shape",       "1x1",
        "--types", "float",   "--metrics-port", port_str.c_str()};
    const auto opts = parse(args);
    ASSERT_TRUE(opts.valid);
    EXPECT_EQ(opts.metrics_port, port);
  }
}
