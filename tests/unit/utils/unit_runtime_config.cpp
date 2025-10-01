#include <gtest/gtest.h>

#include <limits>

#include "utils/runtime_config.hpp"

TEST(RuntimeConfig, ComputeMaxMessageBytesThrowsOnNumelOverflow)
{
  starpu_server::TensorConfig tensor_conf;
  tensor_conf.dims = {std::numeric_limits<int64_t>::max(), 3};
  tensor_conf.type = at::kFloat;
  starpu_server::ModelConfig model;
  model.inputs = {tensor_conf};
  EXPECT_THROW(
      starpu_server::compute_max_message_bytes(1, {model}),
      starpu_server::MessageSizeOverflowException);
}

TEST(RuntimeConfig, ComputeMaxMessageBytesThrowsOnPerSampleBytesOverflow)
{
  starpu_server::TensorConfig tensor_a;
  tensor_a.dims = {
      static_cast<int64_t>(std::numeric_limits<size_t>::max() / 4)};
  tensor_a.type = at::kFloat;

  starpu_server::TensorConfig tensor_b;
  tensor_b.dims = {2};
  tensor_b.type = at::kFloat;
  starpu_server::ModelConfig model;
  model.inputs = {tensor_a};
  model.outputs = {tensor_b};
  EXPECT_THROW(
      starpu_server::compute_max_message_bytes(1, {model}),
      starpu_server::MessageSizeOverflowException);
}

TEST(RuntimeConfig, ComputeModelMessageBytesThrowsOnBatchSizeOverflow)
{
  starpu_server::TensorConfig tensor;
  tensor.dims = {static_cast<int64_t>(std::numeric_limits<size_t>::max() / 4)};
  tensor.type = at::kFloat;

  EXPECT_THROW(
      starpu_server::compute_model_message_bytes(2, {tensor}, {}),
      starpu_server::MessageSizeOverflowException);
}

TEST(RuntimeConfig, ComputeMaxMessageBytesThrowsOnNegativeInputDimension)
{
  starpu_server::TensorConfig tensor_conf;
  tensor_conf.dims = {-1, 2};
  tensor_conf.type = at::kFloat;
  starpu_server::ModelConfig model;
  model.inputs = {tensor_conf};
  EXPECT_THROW(
      starpu_server::compute_max_message_bytes(1, {model}),
      starpu_server::InvalidDimensionException);
}

TEST(RuntimeConfig, ComputeMaxMessageBytesThrowsOnNegativeOutputDimension)
{
  starpu_server::TensorConfig tensor_conf;
  tensor_conf.dims = {2, -1};
  tensor_conf.type = at::kFloat;
  starpu_server::ModelConfig model;
  model.outputs = {tensor_conf};
  EXPECT_THROW(
      starpu_server::compute_max_message_bytes(1, {model}),
      starpu_server::InvalidDimensionException);
}

TEST(RuntimeConfig, ComputeMaxMessageBytesThrowsOnNegativeBatchSize)
{
  starpu_server::TensorConfig tensor_conf2;
  tensor_conf2.dims = {1};
  tensor_conf2.type = at::kFloat;
  starpu_server::ModelConfig model;
  model.inputs = {tensor_conf2};
  EXPECT_THROW(
      starpu_server::compute_max_message_bytes(-1, {model}),
      starpu_server::InvalidDimensionException);
}

TEST(RuntimeConfig, ComputeMaxMessageBytesThrowsOnUnsupportedType)
{
  starpu_server::TensorConfig tensor_conf3;
  tensor_conf3.dims = {1};
  tensor_conf3.type = at::kComplexFloat;
  starpu_server::ModelConfig model;
  model.inputs = {tensor_conf3};
  EXPECT_THROW(
      starpu_server::compute_max_message_bytes(1, {model}),
      starpu_server::UnsupportedDtypeException);
}
