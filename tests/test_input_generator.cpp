#include <gtest/gtest.h>

#include "utils/input_generator.hpp"

using namespace starpu_server::input_generator;

TEST(InputGeneratorUtils, UpperBoundEmbedding)
{
  std::vector<int64_t> shape = {2, 70};
  int64_t result = get_integer_upper_bound(shape, 0);
  EXPECT_EQ(result, BERT_VOCAB_SIZE);
}

TEST(InputGeneratorUtils, UpperBoundDefault)
{
  std::vector<int64_t> shape = {2, 10};
  int64_t result = get_integer_upper_bound(shape, 0);
  EXPECT_EQ(result, DEFAULT_INT_HIGH);
}

TEST(InputGenerator, GeneratesShapesAndTypes)
{
  std::vector<std::vector<int64_t>> shapes = {{2, 3}, {1, 128}};
  std::vector<at::ScalarType> types = {at::kFloat, at::kInt};

  auto tensors = generate_random_inputs(shapes, types);
  ASSERT_EQ(tensors.size(), 2u);

  EXPECT_EQ(tensors[0].sizes(), (torch::IntArrayRef{2, 3}));
  EXPECT_EQ(tensors[0].dtype(), torch::kFloat);

  EXPECT_EQ(tensors[1].sizes(), (torch::IntArrayRef{1, 128}));
  EXPECT_EQ(tensors[1].dtype(), torch::kInt);

  auto max_val = tensors[1].max().item<int64_t>();
  EXPECT_LT(max_val, BERT_VOCAB_SIZE);
}

TEST(InputGenerator, DefaultsToFloatForMissingTypes)
{
  std::vector<std::vector<int64_t>> shapes = {{1, 1}, {2, 2}, {3, 3}};
  std::vector<at::ScalarType> types = {at::kInt};

  auto tensors = generate_random_inputs(shapes, types);

  ASSERT_EQ(tensors.size(), shapes.size());
  EXPECT_EQ(tensors[0].dtype(), torch::kInt);
  EXPECT_EQ(tensors[1].dtype(), torch::kFloat);
  EXPECT_EQ(tensors[2].dtype(), torch::kFloat);
}

TEST(InputGenerator, ThrowsOnUnsupportedType)
{
  std::vector<std::vector<int64_t>> shapes = {{1}};
  std::vector<at::ScalarType> types = {at::kComplexDouble};

  EXPECT_THROW(
      generate_random_inputs(shapes, types),
      starpu_server::UnsupportedDtypeException);
}

TEST(InputGenerator, GeneratesBooleanTensor)
{
  std::vector<std::vector<int64_t>> shapes = {{2, 2}};
  std::vector<at::ScalarType> types = {at::kBool};

  auto tensors = generate_random_inputs(shapes, types);
  ASSERT_EQ(tensors.size(), 1u);

  auto& tensor = tensors[0];
  EXPECT_EQ(tensor.sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(tensor.dtype(), torch::kBool);

  auto min_val = tensor.min().item<uint8_t>();
  auto max_val = tensor.max().item<uint8_t>();
  EXPECT_GE(min_val, 0);
  EXPECT_LE(max_val, 1);
}
