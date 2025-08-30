#include <gtest/gtest.h>
#include <torch/script.h>

#include <stdexcept>

#include "core/starpu_setup.hpp"
#include "utils/exceptions.hpp"

TEST(StarPUSetupErrorsTest, GetCudaWorkersNegativeDeviceThrows)
{
  EXPECT_THROW(
      starpu_server::StarPUSetup::get_cuda_workers_by_device({-1}),
      std::invalid_argument);
}

TEST(StarPUSetupErrorsTest, ExtractTensorsFromOutputUnsupportedType)
{
  constexpr int kAnswer = 42;
  c10::IValue non_tensor{kAnswer};
  EXPECT_THROW(
      starpu_server::extract_tensors_from_output(non_tensor),
      starpu_server::UnsupportedModelOutputTypeException);
}

TEST(StarPUSetupErrorsTest, ExtractTensorsFromOutputDictWithNonTensor)
{
  constexpr int kAnswer = 42;
  c10::impl::GenericDict dict(c10::StringType::get(), c10::AnyType::get());
  dict.insert(c10::IValue("answer"), c10::IValue(kAnswer));

  EXPECT_THROW(
      starpu_server::extract_tensors_from_output(c10::IValue(dict)),
      starpu_server::UnsupportedModelOutputTypeException);
}
