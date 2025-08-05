#include <gtest/gtest.h>

#include <utility>

#include "utils/input_generator.hpp"

using starpu_server::UnsupportedDtypeException;
using starpu_server::input_generator::BERT_VOCAB_SIZE;
using starpu_server::input_generator::DEFAULT_INT_HIGH;
using starpu_server::input_generator::generate_random_inputs;
using starpu_server::input_generator::get_integer_upper_bound;

class UpperBoundTest : public ::testing::TestWithParam<
                           std::pair<std::vector<int64_t>, int64_t>> {};

TEST_P(UpperBoundTest, ReturnsExpectedValue)
{
  const auto& [shape, expected] = GetParam();
  EXPECT_EQ(get_integer_upper_bound(shape, 0), expected);
}

INSTANTIATE_TEST_SUITE_P(
    InputGeneratorUtils, UpperBoundTest,
    ::testing::Values(
        std::make_pair(std::vector<int64_t>{2, 70}, BERT_VOCAB_SIZE),
        std::make_pair(std::vector<int64_t>{2, 10}, DEFAULT_INT_HIGH)));

class InputGeneratorTest : public ::testing::Test {
 protected:
  static std::vector<at::Tensor> generate(
      const std::vector<std::vector<int64_t>>& shapes,
      const std::vector<at::ScalarType>& types)
  {
    return generate_random_inputs(shapes, types);
  }
};

TEST_F(InputGeneratorTest, GeneratesShapesAndTypes)
{
  auto tensors = generate({{2, 3}, {1, 128}}, {at::kFloat, at::kInt});
  ASSERT_EQ(tensors.size(), 2u);
  EXPECT_EQ(tensors[0].sizes(), (torch::IntArrayRef{2, 3}));
  EXPECT_EQ(tensors[0].dtype(), torch::kFloat);
  EXPECT_EQ(tensors[1].sizes(), (torch::IntArrayRef{1, 128}));
  EXPECT_EQ(tensors[1].dtype(), torch::kInt);
  EXPECT_LT(tensors[1].max().item<int64_t>(), BERT_VOCAB_SIZE);
}

TEST_F(InputGeneratorTest, DefaultsToFloatForMissingTypes)
{
  auto tensors = generate({{1, 1}, {2, 2}, {3, 3}}, {at::kInt});
  ASSERT_EQ(tensors.size(), 3u);
  EXPECT_EQ(tensors[0].dtype(), at::kInt);
  EXPECT_EQ(tensors[1].dtype(), at::kFloat);
  EXPECT_EQ(tensors[2].dtype(), at::kFloat);
}

TEST_F(InputGeneratorTest, ThrowsOnUnsupportedType)
{
  EXPECT_THROW(
      generate({{1}}, {at::kComplexDouble}), UnsupportedDtypeException);
}

TEST_F(InputGeneratorTest, GeneratesBooleanTensor)
{
  auto tensors = generate({{2, 2}}, {at::kBool});
  ASSERT_EQ(tensors.size(), 1u);
  const auto& tensor = tensors[0];
  EXPECT_EQ(tensor.sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(tensor.dtype(), at::kBool);
  auto min_val = tensor.min().item<uint8_t>();
  auto max_val = tensor.max().item<uint8_t>();
  EXPECT_GE(min_val, 0);
  EXPECT_LE(max_val, 1);
}
