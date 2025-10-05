#include <gtest/gtest.h>
#include <starpu.h>
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

namespace {
int
failing_stream_query(
    unsigned int /* device_id */, int* /* workerids */,
    enum starpu_worker_archtype /* worker */)
{
  return -1;
}
}  // namespace

TEST(StarPUSetupErrorsTest, GetCudaWorkersQueryFailureThrows)
{
  starpu_server::StarPUSetup::set_worker_stream_query_fn(&failing_stream_query);
  EXPECT_THROW(
      starpu_server::StarPUSetup::get_cuda_workers_by_device({0}),
      starpu_server::StarPUWorkerQueryException);
  starpu_server::StarPUSetup::reset_worker_stream_query_fn();
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
