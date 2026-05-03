#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
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
      const std::optional<starpu_server::ModelConfig> model_opt{model};
      starpu_server::compute_max_message_bytes(batch_size, model_opt);
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
            [](starpu_server::ModelConfig&,
               std::vector<starpu_server::TensorConfig>& inputs,
               std::vector<starpu_server::TensorConfig>&, int64_t&) {
              const size_t type_size = starpu_server::element_size(at::kFloat);
              const size_t threshold =
                  std::numeric_limits<size_t>::max() / type_size;
              const auto dim_size =
                  static_cast<int64_t>(threshold + static_cast<size_t>(1));

              starpu_server::TensorConfig tensor;
              tensor.dims = {dim_size};
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

TEST(RuntimeConfigUtils, ComputeMaxMessageBytesWithoutModelUsesMinimum)
{
  const std::optional<starpu_server::ModelConfig> model;
  EXPECT_EQ(
      starpu_server::compute_max_message_bytes(4, model),
      starpu_server::kDefaultMinMessageBytes);

  constexpr std::size_t kCustomMinBytes = 1234;
  EXPECT_EQ(
      starpu_server::compute_max_message_bytes(4, model, kCustomMinBytes),
      kCustomMinBytes);
}

TEST(
    RuntimeConfigUtils,
    ValidateBatchingSettingsCoherenceRejectsNonPositiveMaxBatchSize)
{
  starpu_server::RuntimeConfig::BatchingSettings batching;
  batching.resolved_max_batch_size = 0;
  batching.pool_size = 1;

  try {
    starpu_server::validate_batching_settings_coherence(batching);
    FAIL() << "Expected validate_batching_settings_coherence to throw.";
  }
  catch (const std::invalid_argument& error) {
    EXPECT_EQ(error.what(), std::string("resolved_max_batch_size must be > 0"));
  }
}

TEST(
    RuntimeConfigUtils,
    ValidateBatchingSettingsCoherenceRejectsNonPositivePoolSize)
{
  starpu_server::RuntimeConfig::BatchingSettings batching;
  batching.resolved_max_batch_size = 1;
  batching.pool_size = 0;

  try {
    starpu_server::validate_batching_settings_coherence(batching);
    FAIL() << "Expected validate_batching_settings_coherence to throw.";
  }
  catch (const std::invalid_argument& error) {
    EXPECT_EQ(error.what(), std::string("pool_size must be > 0"));
  }
}

TEST(
    RuntimeConfigUtils,
    ValidateBatchingSettingsCoherenceIgnoresInactiveAdaptiveSettingsForFixed)
{
  using enum starpu_server::BatchingStrategyKind;

  starpu_server::RuntimeConfig::BatchingSettings batching;
  batching.strategy = Fixed;
  batching.resolved_max_batch_size = 4;
  batching.fixed.batch_size = 4;
  batching.pool_size = 1;
  batching.adaptive.min_batch_size = 3;
  batching.adaptive.max_batch_size = 1;

  EXPECT_NO_THROW(
      starpu_server::validate_batching_settings_coherence(batching));
}

TEST(
    RuntimeConfigUtils,
    ValidateBatchingSettingsCoherenceIgnoresInactiveFixedSettingsForAdaptive)
{
  using enum starpu_server::BatchingStrategyKind;

  starpu_server::RuntimeConfig::BatchingSettings batching;
  batching.strategy = Adaptive;
  batching.resolved_max_batch_size = 4;
  batching.adaptive.min_batch_size = 1;
  batching.adaptive.max_batch_size = 4;
  batching.fixed.batch_size = 0;
  batching.pool_size = 1;

  EXPECT_NO_THROW(
      starpu_server::validate_batching_settings_coherence(batching));
}

TEST(
    RuntimeConfigUtils,
    ValidateBatchingSettingsCoherenceRejectsNonPositiveFixedBatchSizeWhenActive)
{
  using enum starpu_server::BatchingStrategyKind;

  starpu_server::RuntimeConfig::BatchingSettings batching;
  batching.strategy = Fixed;
  batching.resolved_max_batch_size = 1;
  batching.fixed.batch_size = 0;
  batching.pool_size = 1;

  try {
    starpu_server::validate_batching_settings_coherence(batching);
    FAIL() << "Expected validate_batching_settings_coherence to throw.";
  }
  catch (const std::invalid_argument& error) {
    EXPECT_EQ(
        error.what(), std::string("fixed_batching.batch_size must be > 0"));
  }
}

TEST(RuntimeConfigUtils, GpuModelReplicationPolicyToStringHandlesAllBranches)
{
  using starpu_server::GpuModelReplicationPolicy;

  EXPECT_EQ(
      starpu_server::to_string(GpuModelReplicationPolicy::PerDevice),
      "per_device");
  EXPECT_EQ(
      starpu_server::to_string(GpuModelReplicationPolicy::PerWorker),
      "per_worker");

  const auto invalid_policy = static_cast<GpuModelReplicationPolicy>(
      std::numeric_limits<std::uint8_t>::max());
  EXPECT_EQ(starpu_server::to_string(invalid_policy), "per_device");
}

TEST(RuntimeConfigUtils, ParseGpuModelReplicationPolicyParsesKnownValues)
{
  EXPECT_EQ(
      starpu_server::parse_gpu_model_replication_policy("per_device"),
      starpu_server::GpuModelReplicationPolicy::PerDevice);
  EXPECT_EQ(
      starpu_server::parse_gpu_model_replication_policy("per_worker"),
      starpu_server::GpuModelReplicationPolicy::PerWorker);
}

TEST(RuntimeConfigUtils, ParseGpuModelReplicationPolicyRejectsUnknownValue)
{
  EXPECT_THROW(
      starpu_server::parse_gpu_model_replication_policy("invalid_policy"),
      std::invalid_argument);
}

TEST(RuntimeConfigUtils, BatchingStrategyKindToStringHandlesAllBranches)
{
  using starpu_server::BatchingStrategyKind;
  using enum starpu_server::BatchingStrategyKind;

  EXPECT_EQ(starpu_server::to_string(Disabled), "disabled");
  EXPECT_EQ(starpu_server::to_string(Adaptive), "adaptive");
  EXPECT_EQ(starpu_server::to_string(Fixed), "fixed");

  const auto invalid_strategy = static_cast<BatchingStrategyKind>(
      std::numeric_limits<std::uint8_t>::max());
  EXPECT_EQ(starpu_server::to_string(invalid_strategy), "disabled");
}

TEST(RuntimeConfigUtils, ParseBatchingStrategyKindParsesKnownValues)
{
  using enum starpu_server::BatchingStrategyKind;

  EXPECT_EQ(starpu_server::parse_batching_strategy_kind("disabled"), Disabled);
  EXPECT_EQ(starpu_server::parse_batching_strategy_kind("adaptive"), Adaptive);
  EXPECT_EQ(starpu_server::parse_batching_strategy_kind("fixed"), Fixed);
}

TEST(RuntimeConfigUtils, ParseBatchingStrategyKindRejectsUnknownValue)
{
  EXPECT_THROW(
      starpu_server::parse_batching_strategy_kind("invalid_strategy"),
      std::invalid_argument);
}

TEST(RuntimeConfigUtils, DefaultBatchingStrategyIsDisabled)
{
  using enum starpu_server::BatchingStrategyKind;

  starpu_server::RuntimeConfig::BatchingSettings batching;
  EXPECT_EQ(batching.strategy, Disabled);
}

TEST(RuntimeConfigUtils, ResolvedBatchCapacityPrefersEffectiveStrategyLimit)
{
  using enum starpu_server::BatchingStrategyKind;

  starpu_server::RuntimeConfig::BatchingSettings batching;

  batching.strategy = Adaptive;
  batching.resolved_max_batch_size = 4;
  batching.adaptive.max_batch_size = 1;
  EXPECT_EQ(starpu_server::resolved_batch_capacity(batching), 4);

  batching.strategy = Fixed;
  batching.resolved_max_batch_size = 6;
  batching.fixed.batch_size = 1;
  EXPECT_EQ(starpu_server::resolved_batch_capacity(batching), 6);

  batching.strategy = Disabled;
  batching.resolved_max_batch_size = 3;
  EXPECT_EQ(starpu_server::resolved_batch_capacity(batching), 3);
}
