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