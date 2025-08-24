#include <gtest/gtest.h>

#include "utils/input_generator.hpp"

using starpu_server::TensorConfig;
using starpu_server::UnsupportedDtypeException;
using starpu_server::input_generator::generate_random_inputs;

TEST(InputGenerator_Robustesse, ThrowsOnUnsupportedType)
{
  EXPECT_THROW(
      generate_random_inputs({TensorConfig{"", {1}, at::kComplexDouble}}),
      UnsupportedDtypeException);
}
