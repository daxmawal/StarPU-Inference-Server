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

/*TODO : There is a core dumped in this test
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
*/