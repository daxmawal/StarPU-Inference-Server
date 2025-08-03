#include <gtest/gtest.h>

#include "cli/args_parser.hpp"
#include "utils/runtime_config.hpp"

using namespace starpu_server;

TEST(ArgsParser, ParsesRequiredOptions)
{
  std::array<char*, 7> argv = {
      const_cast<char*>("program"),     const_cast<char*>("--model"),
      const_cast<char*>("model.pt"),    const_cast<char*>("--shape"),
      const_cast<char*>("1x3x224x224"), const_cast<char*>("--types"),
      const_cast<char*>("float")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  ASSERT_TRUE(opts.valid);
  EXPECT_EQ(opts.model_path, "model.pt");
  ASSERT_EQ(opts.input_shapes.size(), 1u);
  EXPECT_EQ(opts.input_shapes[0], std::vector<int64_t>({1, 3, 224, 224}));
  ASSERT_EQ(opts.input_types.size(), 1u);
  EXPECT_EQ(opts.input_types[0], at::kFloat);
}

TEST(ArgsParser, MissingRequiredOptions)
{
  std::array<char*, 5> argv = {
      const_cast<char*>("program"), const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shape"),
      const_cast<char*>("1x3x3")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, InvalidNumericValue)
{
  std::array<char*, 9> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shape"),
      const_cast<char*>("1x3x3"),    const_cast<char*>("--types"),
      const_cast<char*>("float"),    const_cast<char*>("--iterations"),
      const_cast<char*>("0")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, ParsesAllOptions)
{
  std::array<char*, 24> argv = {
      const_cast<char*>("program"),
      const_cast<char*>("--model"),
      const_cast<char*>("model.pt"),
      const_cast<char*>("--shapes"),
      const_cast<char*>("1x3x224x224,2x1"),
      const_cast<char*>("--types"),
      const_cast<char*>("float,int"),
      const_cast<char*>("--iterations"),
      const_cast<char*>("5"),
      const_cast<char*>("--device-ids"),
      const_cast<char*>("0,1"),
      const_cast<char*>("--verbose"),
      const_cast<char*>("3"),
      const_cast<char*>("--delay"),
      const_cast<char*>("42"),
      const_cast<char*>("--scheduler"),
      const_cast<char*>("lws"),
      const_cast<char*>("--address"),
      const_cast<char*>("127.0.0.1:1234"),
      const_cast<char*>("--max-msg-size"),
      const_cast<char*>("512"),
      const_cast<char*>("--sync"),
      const_cast<char*>("--no_cpu"),
      nullptr};

  auto args_span = std::span<char*>(argv.data(), argv.size() - 1);
  auto opts = parse_arguments(args_span);

  ASSERT_TRUE(opts.valid);
  EXPECT_EQ(opts.scheduler, "lws");
  EXPECT_EQ(opts.model_path, "model.pt");
  EXPECT_EQ(opts.iterations, 5);
  EXPECT_EQ(opts.delay_ms, 42);
  EXPECT_EQ(opts.verbosity, VerbosityLevel::Debug);
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

TEST(ArgsParser, InvalidShapeString)
{
  std::array<char*, 9> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shape"),
      const_cast<char*>("1x-3x3"),   const_cast<char*>("--types"),
      const_cast<char*>("float")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, InvalidTypeString)
{
  std::array<char*, 9> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shape"),
      const_cast<char*>("1x3"),      const_cast<char*>("--types"),
      const_cast<char*>("unknown")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, InvalidDeviceID)
{
  std::array<char*, 9> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shape"),
      const_cast<char*>("1x3"),      const_cast<char*>("--types"),
      const_cast<char*>("float"),    const_cast<char*>("--device-ids"),
      const_cast<char*>("-1")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, UnknownArgument)
{
  std::array<char*, 8> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shape"),
      const_cast<char*>("1x3"),      const_cast<char*>("--types"),
      const_cast<char*>("float"),    const_cast<char*>("--unknown")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, MismatchedShapesAndTypes)
{
  std::array<char*, 7> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shapes"),
      const_cast<char*>("1x2,2x3"),  const_cast<char*>("--types"),
      const_cast<char*>("float")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, ShapeContainsNonInteger)
{
  std::array<char*, 7> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shape"),
      const_cast<char*>("1xax2"),    const_cast<char*>("--types"),
      const_cast<char*>("float")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, ShapeDimensionOutOfRange)
{
  std::array<char*, 7> argv = {
      const_cast<char*>("program"),
      const_cast<char*>("--model"),
      const_cast<char*>("model.pt"),
      const_cast<char*>("--shape"),
      const_cast<char*>("9223372036854775808"),
      const_cast<char*>("--types"),
      const_cast<char*>("float")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, InvalidVerboseValue)
{
  std::array<char*, 9> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shape"),
      const_cast<char*>("1x3"),      const_cast<char*>("--types"),
      const_cast<char*>("float"),    const_cast<char*>("--verbose"),
      const_cast<char*>("5")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, EmptyShapeString)
{
  std::array<char*, 7> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shape"),
      const_cast<char*>(""),         const_cast<char*>("--types"),
      const_cast<char*>("float")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, EmptyShapesString)
{
  std::array<char*, 7> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shapes"),
      const_cast<char*>(""),         const_cast<char*>("--types"),
      const_cast<char*>("float")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, ShapesTrailingComma)
{
  std::array<char*, 7> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shapes"),
      const_cast<char*>("1x2,"),     const_cast<char*>("--types"),
      const_cast<char*>("float")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, NegativeIterations)
{
  std::array<char*, 9> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shape"),
      const_cast<char*>("1x3"),      const_cast<char*>("--types"),
      const_cast<char*>("float"),    const_cast<char*>("--iterations"),
      const_cast<char*>("-1")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, NegativeDelay)
{
  std::array<char*, 9> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shape"),
      const_cast<char*>("1x3"),      const_cast<char*>("--types"),
      const_cast<char*>("float"),    const_cast<char*>("--delay"),
      const_cast<char*>("-1")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, ZeroMaxMessageSize)
{
  std::array<char*, 9> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shape"),
      const_cast<char*>("1x3"),      const_cast<char*>("--types"),
      const_cast<char*>("float"),    const_cast<char*>("--max-msg-size"),
      const_cast<char*>("0")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, NegativeMaxMessageSize)
{
  std::array<char*, 9> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shape"),
      const_cast<char*>("1x3"),      const_cast<char*>("--types"),
      const_cast<char*>("float"),    const_cast<char*>("--max-msg-size"),
      const_cast<char*>("-1")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, ShapesConsecutiveComma)
{
  std::array<char*, 7> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shapes"),
      const_cast<char*>("1x2,,3"),   const_cast<char*>("--types"),
      const_cast<char*>("float,int")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, MissingModelValue)
{
  std::array<char*, 6> argv = {
      const_cast<char*>("program"), const_cast<char*>("--shape"),
      const_cast<char*>("1x3"),     const_cast<char*>("--types"),
      const_cast<char*>("float"),   const_cast<char*>("--model")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, MissingIterationsValue)
{
  std::array<char*, 8> argv = {
      const_cast<char*>("program"),  const_cast<char*>("--model"),
      const_cast<char*>("model.pt"), const_cast<char*>("--shape"),
      const_cast<char*>("1x3"),      const_cast<char*>("--types"),
      const_cast<char*>("float"),    const_cast<char*>("--iterations")};

  auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

  EXPECT_FALSE(opts.valid);
}

TEST(ArgsParser, VerboseLevels)
{
  using enum VerbosityLevel;
  const std::array<std::pair<const char*, VerbosityLevel>, 5> cases = {{
      {"0", Silent},
      {"1", Info},
      {"2", Stats},
      {"3", Debug},
      {"4", Trace},
  }};

  for (const auto& [level_str, expected] : cases) {
    std::array<char*, 9> argv = {
        const_cast<char*>("program"),  const_cast<char*>("--model"),
        const_cast<char*>("model.pt"), const_cast<char*>("--shape"),
        const_cast<char*>("1x3"),      const_cast<char*>("--types"),
        const_cast<char*>("float"),    const_cast<char*>("--verbose"),
        const_cast<char*>(level_str)};

    auto opts = parse_arguments(std::span<char*>(argv.data(), argv.size()));

    ASSERT_TRUE(opts.valid);
    EXPECT_EQ(opts.verbosity, expected);
  }
}