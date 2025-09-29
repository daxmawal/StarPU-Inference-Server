#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include "test_helpers.hpp"
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
  yaml << "dynamic_batching: true\n";
  yaml << "pregen_inputs: 8\n";
  yaml << "warmup_pregen_inputs: 5\n";
  yaml << "warmup_iterations: 3\n";
  yaml << "seed: 123\n";
  yaml << "validate_results: false\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_valid.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.scheduler, "fcfs");
  EXPECT_EQ(cfg.models[0].path, model_path.string());
  EXPECT_EQ(cfg.device_ids, (std::vector<int>{0, 1}));
  ASSERT_EQ(cfg.models[0].inputs.size(), 1U);
  EXPECT_EQ(cfg.models[0].inputs[0].name, "in");
  EXPECT_EQ(
      cfg.models[0].inputs[0].dims, (std::vector<int64_t>{1, 3, 224, 224}));
  EXPECT_EQ(cfg.models[0].inputs[0].type, at::kFloat);
  ASSERT_EQ(cfg.models[0].outputs.size(), 1U);
  EXPECT_EQ(cfg.models[0].outputs[0].name, "out");
  EXPECT_EQ(cfg.models[0].outputs[0].dims, (std::vector<int64_t>{1, 1000}));
  EXPECT_EQ(cfg.models[0].outputs[0].type, at::kFloat);
  EXPECT_EQ(cfg.verbosity, VerbosityLevel::Debug);
  EXPECT_EQ(cfg.max_batch_size, 4);
  EXPECT_TRUE(cfg.dynamic_batching);
  EXPECT_EQ(cfg.pregen_inputs, 8U);
  EXPECT_EQ(cfg.warmup_pregen_inputs, 5U);
  EXPECT_EQ(cfg.warmup_iterations, 3);
  const bool has_seed = cfg.seed.has_value();
  ASSERT_TRUE(has_seed);
  const auto seed_value = cfg.seed.value_or(0U);
  EXPECT_EQ(seed_value, 123U);
  EXPECT_FALSE(cfg.validate_results);
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
  const auto model_path = std::filesystem::temp_directory_path() /
                          "config_loader_neg_delay_model.pt";
  std::ofstream(model_path).put('\0');

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
  yaml << "delay: -10\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_neg_delay.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, ParsesDelayAndAddress)
{
  const auto model_path = std::filesystem::temp_directory_path() /
                          "config_loader_delay_addr_model.pt";
  std::ofstream(model_path).put('\0');

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
  yaml << "delay: 15\n";
  yaml << "address: 127.0.0.1:50051\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_delay_addr.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.delay_ms, 15);
  EXPECT_EQ(cfg.server_address, "127.0.0.1:50051");
}

TEST(ConfigLoader, NegativeIterationsSetsValidFalse)
{
  const auto model_path = std::filesystem::temp_directory_path() /
                          "config_loader_neg_iter_model.pt";
  std::ofstream(model_path).put('\0');

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
  yaml << "iterations: -1\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_neg_iter.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, MetricsPortOutOfRangeSetsValidFalse)
{
  const auto model_path = std::filesystem::temp_directory_path() /
                          "config_loader_bad_metrics_port_model.pt";
  std::ofstream(model_path).put('\0');

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
  yaml << "metrics_port: 70000\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_bad_metrics_port.yaml";
  std::ofstream(tmp) << yaml.str();

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

  const auto tmp = std::filesystem::temp_directory_path() /
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

TEST(ConfigLoader, DimensionExceedsIntMaxSetsValidFalse)
{
  const auto model_path = std::filesystem::temp_directory_path() /
                          "config_loader_large_dim_model.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "model: " << model_path.string() << "\n";
  yaml << "input:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1, "
       << static_cast<long long>(std::numeric_limits<int>::max()) + 1 << "]\n";
  yaml << "    data_type: float32\n";
  yaml << "output:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_large_dim.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, MissingDimsSetsValidFalse)
{
  const std::string yaml = R"(
model: model.pt
input:
  - name: in
    data_type: float32
output:
  - name: out
    dims: [1]
    data_type: float32
)";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_missing_dims.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, MissingDataTypeSetsValidFalse)
{
  const std::string yaml = R"(
model: model.pt
input:
  - name: in
    dims: [1]
output:
  - name: out
    dims: [1]
    data_type: float32
)";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_missing_dtype.yaml";
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

TEST(ConfigLoader, TooManyInputsSetsValidFalse)
{
  const auto model_path =
      std::filesystem::temp_directory_path() / "config_loader_many_inputs.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "model: " << model_path.string() << "\n";
  yaml << "input:\n";
  for (std::size_t i = 0; i <= kMaxInputs; ++i) {
    yaml << "  - name: in" << i << "\n";
    yaml << "    dims: [1]\n";
    yaml << "    data_type: float32\n";
  }
  yaml << "output:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_too_many_inputs.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, TooManyDimsSetsValidFalse)
{
  const auto model_path =
      std::filesystem::temp_directory_path() / "config_loader_many_dims.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "model: " << model_path.string() << "\n";
  yaml << "input:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [";
  for (std::size_t i = 0; i <= kMaxDims; ++i) {
    if (i != 0U) {
      yaml << ", ";
    }
    yaml << 1;
  }
  yaml << "]\n";
  yaml << "    data_type: float32\n";
  yaml << "output:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_too_many_dims.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
}

TEST(
    ConfigLoader,
    MessageSizeOverflowDuringMaxMessageComputationMarksConfigInvalid)
{
  const auto model_path = std::filesystem::temp_directory_path() /
                          "config_loader_overflow_dims_model.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "model: " << model_path.string() << "\n";
  yaml << "input:\n";
  yaml << "  - name: huge\n";
  yaml << "    dims: [2147483647, 2147483647, 2147483647]\n";
  yaml << "    data_type: float32\n";
  yaml << "output:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_overflow_dims.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: numel * dimension size would overflow size_t";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
  ASSERT_EQ(cfg.models.size(), 1U);
  ASSERT_EQ(cfg.models[0].inputs.size(), 1U);
  EXPECT_EQ(
      cfg.models[0].inputs[0].dims,
      (std::vector<int64_t>{2147483647, 2147483647, 2147483647}));
  ASSERT_EQ(cfg.models[0].outputs.size(), 1U);

  // InvalidDimensionException cannot be triggered here because
  // parse_tensor_nodes already rejects non-positive dimensions when reading the
  // YAML. Covering the overflow path protects compute_max_message_bytes against
  // future changes.
}

TEST(ConfigLoader, UnsupportedDtypeDuringMaxMessageComputationMarksInvalid)
{
  const auto model_path = std::filesystem::temp_directory_path() /
                          "config_loader_complex_dtype_model.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "model: " << model_path.string() << "\n";
  yaml << "input:\n";
  yaml << "  - name: complex_input\n";
  yaml << "    dims: [1, 1]\n";
  yaml << "    data_type: complex64\n";
  yaml << "output:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_complex_dtype.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: Unsupported at::ScalarType";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
  ASSERT_EQ(cfg.models.size(), 1U);
  ASSERT_EQ(cfg.models[0].inputs.size(), 1U);
  EXPECT_EQ(
      cfg.models[0].inputs[0].type,
      starpu_server::string_to_scalar_type("complex64"));
  ASSERT_EQ(cfg.models[0].outputs.size(), 1U);
  EXPECT_FALSE(cfg.models[0].outputs.empty());
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
