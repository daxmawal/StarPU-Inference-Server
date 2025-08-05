#include <gtest/gtest.h>

#include <vector>

#include "cli/args_parser.hpp"
#include "cli_test_utils.hpp"
#include "utils/runtime_config.hpp"

TEST(ArgsParser, ParsesRequiredOptions)
{
  const auto opts = parse(
      {"program", "--model", "model.pt", "--shape", "1x3x224x224", "--types",
       "float"});
  ASSERT_TRUE(opts.valid);
  EXPECT_EQ(opts.model_path, "model.pt");
  ASSERT_EQ(opts.input_shapes.size(), 1u);
  EXPECT_EQ(opts.input_shapes[0], std::vector<int64_t>({1, 3, 224, 224}));
  ASSERT_EQ(opts.input_types.size(), 1u);
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
  ASSERT_EQ(opts.input_shapes.size(), 2u);
  EXPECT_EQ(opts.input_shapes[0], std::vector<int64_t>({1, 3, 224, 224}));
  EXPECT_EQ(opts.input_shapes[1], std::vector<int64_t>({2, 1}));
  ASSERT_EQ(opts.input_types.size(), 2u);
  EXPECT_EQ(opts.input_types[0], at::kFloat);
  EXPECT_EQ(opts.input_types[1], at::kInt);
}


TEST(ArgsParser, MissingRequiredOptions)
{
  expect_invalid({"program", "--model", "model.pt", "--shape", "1x3x3"});
}

TEST(ArgsParser, InvalidNumericValue)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shape", "1x3x3", "--types", "float",
       "--iterations", "0"});
}

TEST(ArgsParser, InvalidShapeString)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shape", "1x-3x3", "--types",
       "float"});
}

TEST(ArgsParser, InvalidTypeString)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shape", "1x3", "--types",
       "unknown"});
}

TEST(ArgsParser, InvalidDeviceID)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shape", "1x3", "--types", "float",
       "--device-ids", "-1"});
}

TEST(ArgsParser, UnknownArgument)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shape", "1x3", "--types", "float",
       "--unknown"});
}

TEST(ArgsParser, MismatchedShapesAndTypes)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shapes", "1x2,2x3", "--types",
       "float"});
}

TEST(ArgsParser, ShapeContainsNonInteger)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shape", "1xax2", "--types",
       "float"});
}

TEST(ArgsParser, ShapeDimensionOutOfRange)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shape", "9223372036854775808",
       "--types", "float"});
}

TEST(ArgsParser, InvalidVerboseValue)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shape", "1x3", "--types", "float",
       "--verbose", "5"});
}

TEST(ArgsParser, EmptyShapeString)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shape", "", "--types", "float"});
}

TEST(ArgsParser, EmptyShapesString)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shapes", "", "--types", "float"});
}

TEST(ArgsParser, ShapesTrailingComma)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shapes", "1x2,", "--types",
       "float"});
}

TEST(ArgsParser, NegativeIterations)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shape", "1x3", "--types", "float",
       "--iterations", "-1"});
}

TEST(ArgsParser, NegativeDelay)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shape", "1x3", "--types", "float",
       "--delay", "-1"});
}

TEST(ArgsParser, ZeroMaxMessageSize)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shape", "1x3", "--types", "float",
       "--max-msg-size", "0"});
}

TEST(ArgsParser, NegativeMaxMessageSize)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shape", "1x3", "--types", "float",
       "--max-msg-size", "-1"});
}

TEST(ArgsParser, ShapesConsecutiveComma)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shapes", "1x2,,3", "--types",
       "float,int"});
}

TEST(ArgsParser, MissingModelValue)
{
  expect_invalid({"program", "--shape", "1x3", "--types", "float", "--model"});
}

TEST(ArgsParser, MissingIterationsValue)
{
  expect_invalid(
      {"program", "--model", "model.pt", "--shape", "1x3", "--types", "float",
       "--iterations"});
}

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
