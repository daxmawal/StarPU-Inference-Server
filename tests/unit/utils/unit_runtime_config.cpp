#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

#include "utils/runtime_config.hpp"

namespace {

struct RuntimeErrorCase {
  std::function<void(
      starpu_server::ModelConfig& model,
      std::vector<starpu_server::TensorConfig>& inputs,
      std::vector<starpu_server::TensorConfig>& outputs, int64_t& batch_size)>
      prepare_configs;
  std::function<void(const std::function<void()>&)> expect_exception;
  std::function<void(
      int64_t batch_size, const starpu_server::ModelConfig& model,
      const std::vector<starpu_server::TensorConfig>& inputs,
      const std::vector<starpu_server::TensorConfig>& outputs)>
      callable;
};

template <typename Exception>
auto
make_expect_throw() -> std::function<void(const std::function<void()>&)>
{
  return [](const std::function<void()>& invocation) {
    EXPECT_THROW(invocation(), Exception);
  };
}

const auto compute_max_message_bytes_callable =
    [](int64_t batch_size, const starpu_server::ModelConfig& model,
       const std::vector<starpu_server::TensorConfig>&,
       const std::vector<starpu_server::TensorConfig>&) {
      std::vector<starpu_server::ModelConfig> models{model};
      starpu_server::compute_max_message_bytes(batch_size, models);
    };

const auto compute_model_message_bytes_callable =
    [](int64_t batch_size, const starpu_server::ModelConfig&,
       const std::vector<starpu_server::TensorConfig>& inputs,
       const std::vector<starpu_server::TensorConfig>& outputs) {
      starpu_server::compute_model_message_bytes(batch_size, inputs, outputs);
    };

class RuntimeConfigErrorTest
    : public ::testing::TestWithParam<RuntimeErrorCase> {};

TEST_P(RuntimeConfigErrorTest, ThrowsExpectedException)
{
  const auto& param = GetParam();

  int64_t batch_size = 1;
  starpu_server::ModelConfig model;
  std::vector<starpu_server::TensorConfig> inputs;
  std::vector<starpu_server::TensorConfig> outputs;

  param.prepare_configs(model, inputs, outputs, batch_size);

  const auto invocation = [&]() {
    param.callable(batch_size, model, inputs, outputs);
  };

  param.expect_exception(invocation);
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    RuntimeConfigExceptions, RuntimeConfigErrorTest,
    ::testing::Values(
        RuntimeErrorCase{
            [](starpu_server::ModelConfig& model,
               std::vector<starpu_server::TensorConfig>&,
               std::vector<starpu_server::TensorConfig>&, int64_t&) {
              starpu_server::TensorConfig tensor_conf;
              tensor_conf.dims = {std::numeric_limits<int64_t>::max(), 3};
              tensor_conf.type = at::kFloat;
              model.inputs = {tensor_conf};
            },
            make_expect_throw<starpu_server::MessageSizeOverflowException>(),
            compute_max_message_bytes_callable},
        RuntimeErrorCase{
            [](starpu_server::ModelConfig& model,
               std::vector<starpu_server::TensorConfig>&,
               std::vector<starpu_server::TensorConfig>&, int64_t&) {
              starpu_server::TensorConfig tensor_a;
              tensor_a.dims = {
                  static_cast<int64_t>(std::numeric_limits<size_t>::max() / 4)};
              tensor_a.type = at::kFloat;

              starpu_server::TensorConfig tensor_b;
              tensor_b.dims = {2};
              tensor_b.type = at::kFloat;

              model.inputs = {tensor_a};
              model.outputs = {tensor_b};
            },
            make_expect_throw<starpu_server::MessageSizeOverflowException>(),
            compute_max_message_bytes_callable},
        RuntimeErrorCase{
            [](starpu_server::ModelConfig&,
               std::vector<starpu_server::TensorConfig>& inputs,
               std::vector<starpu_server::TensorConfig>&, int64_t& batch_size) {
              batch_size = 2;

              starpu_server::TensorConfig tensor;
              tensor.dims = {
                  static_cast<int64_t>(std::numeric_limits<size_t>::max() / 4)};
              tensor.type = at::kFloat;

              inputs = {tensor};
            },
            make_expect_throw<starpu_server::MessageSizeOverflowException>(),
            compute_model_message_bytes_callable},
        RuntimeErrorCase{
            [](starpu_server::ModelConfig& model,
               std::vector<starpu_server::TensorConfig>&,
               std::vector<starpu_server::TensorConfig>&, int64_t&) {
              starpu_server::TensorConfig tensor_conf;
              tensor_conf.dims = {-1, 2};
              tensor_conf.type = at::kFloat;
              model.inputs = {tensor_conf};
            },
            make_expect_throw<starpu_server::InvalidDimensionException>(),
            compute_max_message_bytes_callable},
        RuntimeErrorCase{
            [](starpu_server::ModelConfig& model,
               std::vector<starpu_server::TensorConfig>&,
               std::vector<starpu_server::TensorConfig>&, int64_t&) {
              starpu_server::TensorConfig tensor_conf;
              tensor_conf.dims = {2, -1};
              tensor_conf.type = at::kFloat;
              model.outputs = {tensor_conf};
            },
            make_expect_throw<starpu_server::InvalidDimensionException>(),
            compute_max_message_bytes_callable},
        RuntimeErrorCase{
            [](starpu_server::ModelConfig& model,
               std::vector<starpu_server::TensorConfig>&,
               std::vector<starpu_server::TensorConfig>&, int64_t& batch_size) {
              batch_size = -1;

              starpu_server::TensorConfig tensor_conf;
              tensor_conf.dims = {1};
              tensor_conf.type = at::kFloat;
              model.inputs = {tensor_conf};
            },
            make_expect_throw<starpu_server::InvalidDimensionException>(),
            compute_max_message_bytes_callable},
        RuntimeErrorCase{
            [](starpu_server::ModelConfig& model,
               std::vector<starpu_server::TensorConfig>&,
               std::vector<starpu_server::TensorConfig>&, int64_t&) {
              starpu_server::TensorConfig tensor_conf;
              tensor_conf.dims = {1};
              tensor_conf.type = at::kComplexFloat;
              model.inputs = {tensor_conf};
            },
            make_expect_throw<starpu_server::UnsupportedDtypeException>(),
            compute_max_message_bytes_callable}));
