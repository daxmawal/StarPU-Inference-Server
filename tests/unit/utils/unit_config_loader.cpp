#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <vector>

#include "utils/config_loader.hpp"

using namespace starpu_server;

TEST(ConfigLoader, LoadsValidConfig)
{
  const std::string yaml = R"(
scheduler: fcfs
model: model.pt
device_ids: [0, 1]
input:
  - name: in
    dims: [1, 3, 224, 224]
    data_type: float32
output:
  - name: out
    dims: [1, 1000]
    data_type: float32
verbosity: 3
max_batch_size: 4
pregen_inputs: 8
warmup_iterations: 3
)";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_valid.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.scheduler, "fcfs");
  EXPECT_EQ(cfg.model_path, "model.pt");
  EXPECT_EQ(cfg.device_ids, (std::vector<int>{0, 1}));
  ASSERT_EQ(cfg.inputs.size(), 1U);
  EXPECT_EQ(cfg.inputs[0].name, "in");
  EXPECT_EQ(cfg.inputs[0].dims, (std::vector<int64_t>{1, 3, 224, 224}));
  EXPECT_EQ(cfg.inputs[0].type, at::kFloat);
  ASSERT_EQ(cfg.outputs.size(), 1U);
  EXPECT_EQ(cfg.outputs[0].name, "out");
  EXPECT_EQ(cfg.outputs[0].dims, (std::vector<int64_t>{1, 1000}));
  EXPECT_EQ(cfg.outputs[0].type, at::kFloat);
  EXPECT_EQ(cfg.verbosity, VerbosityLevel::Debug);
  EXPECT_EQ(cfg.max_batch_size, 4);
  EXPECT_EQ(cfg.pregen_inputs, 8U);
  EXPECT_EQ(cfg.warmup_iterations, 3);
  EXPECT_TRUE(cfg.use_cuda);
}

TEST(ConfigLoader, InvalidConfigSetsValidFalse)
{
  const std::string yaml = R"(max_batch_size: 0)";
  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_invalid.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
}

#define load_config load_config_unused
#include "utils/config_loader.cpp"
#undef load_config

using AliasCase =
    ::testing::TestWithParam<std::pair<const char*, at::ScalarType>>;

TEST_P(AliasCase, ParsesAliasesToExpectedScalarType)
{
  const auto& [alias, expected] = GetParam();
  EXPECT_EQ(starpu_server::parse_type_string(alias), expected);
}

INSTANTIATE_TEST_SUITE_P(
    ValidAliases, AliasCase,
    ::testing::Values(
        std::pair{"float", at::kFloat}, std::pair{"float32", at::kFloat},
        std::pair{"TYPE_FP16", at::kHalf}, std::pair{"bf16", at::kBFloat16},
        std::pair{"INT64", at::kLong}, std::pair{"bool", at::kBool}));

TEST(ParseTypeString, UnknownTypeThrows)
{
  EXPECT_THROW(
      starpu_server::parse_type_string("unknown_type"), std::invalid_argument);
}
