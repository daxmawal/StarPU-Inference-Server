#include <gtest/gtest.h>

#include "utils/input_generator.hpp"

using namespace starpu_server::input_generator;

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
