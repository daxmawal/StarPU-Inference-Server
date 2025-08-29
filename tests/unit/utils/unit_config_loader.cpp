#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include "utils/config_loader.hpp"
#include "utils/datatype_utils.hpp"

using namespace starpu_server;

TEST(ConfigLoader, LoadsValidConfig)
{
  const auto model_path =
      std::filesystem::temp_directory_path() / "config_loader_valid_model.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "scheduler: fcfs\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "device_ids: [0, 1]\n";
  yaml << "input:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1, 3, 224, 224]\n";
  yaml << "    data_type: float32\n";
  yaml << "output:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1, 1000]\n";
  yaml << "    data_type: float32\n";
  yaml << "verbosity: 3\n";
  yaml << "max_batch_size: 4\n";
  yaml << "pregen_inputs: 8\n";
  yaml << "warmup_pregen_inputs: 5\n";
  yaml << "warmup_iterations: 3\n";
  yaml << "seed: 123\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_valid.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.scheduler, "fcfs");
  EXPECT_EQ(cfg.model_path, model_path.string());
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
  EXPECT_EQ(cfg.warmup_pregen_inputs, 5U);
  EXPECT_EQ(cfg.warmup_iterations, 3);
  ASSERT_TRUE(cfg.seed.has_value());
  EXPECT_EQ(cfg.seed.value(), 123U);
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

TEST(ConfigLoader, NegativeDelaySetsValidFalse)
{
  const std::string yaml = R"(delay: -10)";
  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_neg_delay.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, InvalidSchedulerSetsValidFalse)
{
  const auto model_path = std::filesystem::temp_directory_path() /
                          "config_loader_invalid_sched_model.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "scheduler: unknown\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "input:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "output:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";

  const auto tmp =
      std::filesystem::temp_directory_path() /
      "config_loader_invalid_sched.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, NegativeDimensionSetsValidFalse)
{
  const std::string yaml = R"(
model: model.pt
input:
  - name: in
    dims: [-1, 3]
    data_type: float32
output:
  - name: out
    dims: [1]
    data_type: float32
)";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_neg_dim.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, ZeroDimensionSetsValidFalse)
{
  const std::string yaml = R"(
model: model.pt
input:
  - name: in
    dims: [0, 3]
    data_type: float32
output:
  - name: out
    dims: [1]
    data_type: float32
)";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_zero_dim.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, InvalidVerbositySetsValidFalse)
{
  const std::string yaml = R"(verbosity: unknown)";
  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_invalid_verbosity.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, MissingModelSetsValidFalse)
{
  const std::string yaml = R"(
input:
  - name: in
    dims: [1]
    data_type: float32
output:
  - name: out
    dims: [1]
    data_type: float32
)";
  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_no_model.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, NonexistentModelFileSetsValidFalse)
{
  const auto model_path =
      std::filesystem::temp_directory_path() / "nonexistent_model.pt";
  std::ostringstream yaml;
  yaml << "model: " << model_path.string() << "\n";
  yaml << "input:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "output:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_missing_model_file.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, MissingModelSkipsParsingOtherKeys)
{
  const std::string yaml = R"(
input:
  - name: in
    dims: [1]
    data_type: float32
output:
  - name: out
    dims: [1]
    data_type: float32
max_batch_size: 0
)";
  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_no_model_skip.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(cfg.max_batch_size, 1);
}

TEST(ConfigLoader, MissingInputSetsValidFalse)
{
  const std::string yaml = R"(
model: model.pt
output:
  - name: out
    dims: [1]
    data_type: float32
)";
  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_no_input.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, MissingInputSkipsParsingOtherKeys)
{
  const std::string yaml = R"(
model: model.pt
output:
  - name: out
    dims: [1]
    data_type: float32
delay: -10
)";
  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_no_input_skip.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(cfg.delay_ms, 0);
}

TEST(ConfigLoader, MissingOutputSetsValidFalse)
{
  const std::string yaml = R"(
model: model.pt
input:
  - name: in
    dims: [1]
    data_type: float32
)";
  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_no_output.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, MissingOutputSkipsParsingOtherKeys)
{
  const std::string yaml = R"(
model: model.pt
input:
  - name: in
    dims: [1]
    data_type: float32
max_batch_size: 0
)";
  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_no_output_skip.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(cfg.max_batch_size, 1);
}

using VerbosityCase =
    ::testing::TestWithParam<std::pair<const char*, VerbosityLevel>>;

TEST_P(VerbosityCase, ParsesVerbosityStrings)
{
  const auto& [value, expected] = GetParam();
  const std::string yaml = std::string("verbosity: ") + value;
  const auto tmp = std::filesystem::temp_directory_path() /
                   (std::string("config_loader_verbosity_") + value + ".yaml");
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_EQ(cfg.verbosity, expected);
}

INSTANTIATE_TEST_SUITE_P(
    VerbosityLevels, VerbosityCase,
    ::testing::Values(
        std::pair{"silent", VerbosityLevel::Silent},
        std::pair{"info", VerbosityLevel::Info},
        std::pair{"stats", VerbosityLevel::Stats},
        std::pair{"debug", VerbosityLevel::Debug},
        std::pair{"trace", VerbosityLevel::Trace},
        std::pair{"TrAcE", VerbosityLevel::Trace}));

using AliasCase =
    ::testing::TestWithParam<std::pair<const char*, at::ScalarType>>;

TEST_P(AliasCase, ParsesAliasesToExpectedScalarType)
{
  const auto& [alias, expected] = GetParam();
  EXPECT_EQ(starpu_server::string_to_scalar_type(alias), expected);
}

INSTANTIATE_TEST_SUITE_P(
    ValidAliases, AliasCase,
    ::testing::Values(
        std::pair{"float", at::kFloat}, std::pair{"float32", at::kFloat},
        std::pair{"TYPE_FP16", at::kHalf}, std::pair{"bf16", at::kBFloat16},
        std::pair{"INT64", at::kLong}, std::pair{"bool", at::kBool}));

TEST(StringToScalarType, UnknownTypeThrows)
{
  EXPECT_THROW(
      starpu_server::string_to_scalar_type("unknown_type"),
      std::invalid_argument);
}
