#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>

#include "test_helpers.hpp"
#include "utils/config_loader.hpp"
#include "utils/datatype_utils.hpp"

using namespace starpu_server;

namespace {

auto
WriteTempFile(const std::string& name, const std::string& contents)
    -> std::filesystem::path
{
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream(path) << contents;
  return path;
}

struct ScopedPermissionRestorer {
  explicit ScopedPermissionRestorer(std::filesystem::path p)
      : path(std::move(p))
  {
  }

  ~ScopedPermissionRestorer()
  {
    std::error_code ec;
    std::filesystem::permissions(
        path, std::filesystem::perms::owner_all,
        std::filesystem::perm_options::replace, ec);
    std::filesystem::remove_all(path, ec);
  }

  std::filesystem::path path;
};

auto
MakeUniqueTempDir(const std::string& prefix) -> std::filesystem::path
{
  const auto unique_suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
  auto path = std::filesystem::temp_directory_path() /
              (prefix + "_" + std::to_string(unique_suffix));
  std::filesystem::create_directories(path);
  return path;
}

struct ConfigLoaderHookGuard {
  explicit ConfigLoaderHookGuard(ConfigLoaderPostParseHook hook)
  {
    set_config_loader_post_parse_hook(std::move(hook));
  }

  ~ConfigLoaderHookGuard() { reset_config_loader_post_parse_hook(); }
};

auto
base_model_yaml() -> std::string
{
  return std::string{
      "name: config_loader_test\n"
      "model: {{MODEL_PATH}}\n"
      "inputs:\n"
      "  - name: in\n"
      "    dims: [1]\n"
      "    data_type: float32\n"
      "outputs:\n"
      "  - name: out\n"
      "    dims: [1]\n"
      "    data_type: float32\n"
      "batch_coalesce_timeout_ms: 1\n"
      "max_batch_size: 1\n"
      "pool_size: 1\n"};
}

struct InvalidConfigCase {
  std::string name;
  std::string yaml;
  std::optional<std::string> expected_error;
  bool needs_model_path = true;
  bool create_model_file = true;
};

class InvalidConfigTest : public ::testing::TestWithParam<InvalidConfigCase> {};

auto
InvalidConfigCaseName(const ::testing::TestParamInfo<InvalidConfigCase>& info)
    -> std::string
{
  return info.param.name;
}

const std::vector<InvalidConfigCase> kInvalidConfigCases = {
    InvalidConfigCase{
        "UnknownKeySetsValidFalse",
        [] {
          auto yaml = base_model_yaml();
          yaml += "unknown_option: true\n";
          return yaml;
        }(),
        "Unknown configuration option: unknown_option"},
    InvalidConfigCase{
        "NonScalarKeySetsValidFalse",
        [] {
          auto yaml = base_model_yaml();
          yaml += "? [invalid, key]\n";
          yaml += ": true\n";
          return yaml;
        }(),
        "Configuration keys must be scalar strings"},
    InvalidConfigCase{
        "DeviceIdsAtRootInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "device_ids: [0]\n";
          return yaml;
        }(),
        "device_ids must be nested inside the use_cuda block (e.g. "
        "\"use_cuda: [{ device_ids: [0] }]\")"},
    InvalidConfigCase{
        "InvalidConfigSetsValidFalse",
        [] {
          auto yaml = base_model_yaml();
          yaml += "max_batch_size: 0\n";
          return yaml;
        }(),
        std::nullopt, false, false},
    InvalidConfigCase{
        "NegativeDelaySetsValidFalse",
        [] {
          auto yaml = base_model_yaml();
          yaml += "delay_us: -10\n";
          return yaml;
        }(),
        std::nullopt},
    InvalidConfigCase{
        "UseCudaEmptySequenceInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "use_cuda: []\n";
          return yaml;
        }(),
        "use_cuda requires at least one device_ids entry"},
    InvalidConfigCase{
        "UseCudaNonSequenceInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "use_cuda:\n";
          yaml += "  device_ids: [0]\n";
          return yaml;
        }(),
        "use_cuda must be a boolean or a sequence of device mappings"},
    InvalidConfigCase{
        "UseCudaEntryNotMapInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "use_cuda:\n";
          yaml += "  - true\n";
          yaml += "  - { device_ids: [0] }\n";
          return yaml;
        }(),
        "use_cuda entries must be mappings that define device_ids"},
    InvalidConfigCase{
        "UseCudaEntryMissingDeviceIdsInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "use_cuda:\n";
          yaml += "  - {}\n";
          yaml += "  - { device_ids: [0] }\n";
          return yaml;
        }(),
        "use_cuda entries require a device_ids sequence"},
    InvalidConfigCase{
        "UseCudaEntryDeviceIdsNotSequenceInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "use_cuda:\n";
          yaml += "  - { device_ids: 0 }\n";
          yaml += "  - { device_ids: [1] }\n";
          return yaml;
        }(),
        "device_ids inside use_cuda must be a sequence"},
    InvalidConfigCase{
        "StarpuEnvNotMapInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "starpu_env: []\n";
          return yaml;
        }(),
        "starpu_env must be a mapping of variable names to values"},
    InvalidConfigCase{
        "StarpuEnvKeyNotScalarInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "starpu_env:\n";
          yaml += "  ? [invalid, key]\n";
          yaml += "  : value\n";
          return yaml;
        }(),
        "starpu_env entries must have scalar keys"},
    InvalidConfigCase{
        "StarpuEnvValueNotScalarInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "starpu_env:\n";
          yaml += "  VAR: [1, 2]\n";
          return yaml;
        }(),
        "starpu_env entries must have scalar values"},
    InvalidConfigCase{
        "NegativeBatchCoalesceTimeoutSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name: config_loader_test\n";
          yaml += "model: {{MODEL_PATH}}\n";
          yaml += "inputs:\n";
          yaml += "  - name: in\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "outputs:\n";
          yaml += "  - name: out\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "max_batch_size: 1\n";
          yaml += "pool_size: 1\n";
          yaml += "batch_coalesce_timeout_ms: -5\n";
          return yaml;
        }(),
        std::nullopt},
    InvalidConfigCase{
        "NegativeRequestNbSetsValidFalse",
        [] {
          auto yaml = base_model_yaml();
          yaml += "request_nb: -1\n";
          return yaml;
        }(),
        std::nullopt},
    InvalidConfigCase{
        "MetricsPortOutOfRangeSetsValidFalse",
        [] {
          auto yaml = base_model_yaml();
          yaml += "metrics_port: 70000\n";
          return yaml;
        }(),
        std::nullopt},
    InvalidConfigCase{
        "InvalidSchedulerSetsValidFalse",
        [] {
          auto yaml = std::string{"scheduler: unknown\n"};
          yaml += base_model_yaml();
          return yaml;
        }(),
        std::nullopt},
    InvalidConfigCase{
        "NegativeDimensionSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name: config_loader_test\n";
          yaml += "model: {{MODEL_PATH}}\n";
          yaml += "inputs:\n";
          yaml += "  - name: in\n";
          yaml += "    dims: [-1, 3]\n";
          yaml += "    data_type: float32\n";
          yaml += "outputs:\n";
          yaml += "  - name: out\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "batch_coalesce_timeout_ms: 1\n";
          yaml += "max_batch_size: 1\n";
          yaml += "pool_size: 1\n";
          return yaml;
        }(),
        std::nullopt},
    InvalidConfigCase{
        "ZeroDimensionSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name: config_loader_test\n";
          yaml += "model: {{MODEL_PATH}}\n";
          yaml += "inputs:\n";
          yaml += "  - name: in\n";
          yaml += "    dims: [0, 3]\n";
          yaml += "    data_type: float32\n";
          yaml += "outputs:\n";
          yaml += "  - name: out\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "batch_coalesce_timeout_ms: 1\n";
          yaml += "max_batch_size: 1\n";
          yaml += "pool_size: 1\n";
          return yaml;
        }(),
        std::nullopt},
    InvalidConfigCase{
        "DimensionExceedsIntMaxSetsValidFalse",
        [] {
          std::ostringstream yaml;
          yaml << "name: config_loader_test\n";
          yaml << "model: {{MODEL_PATH}}\n";
          yaml << "inputs:\n";
          yaml << "  - name: in\n";
          yaml << "    dims: [1, "
               << static_cast<long long>(std::numeric_limits<int>::max()) + 1
               << "]\n";
          yaml << "    data_type: float32\n";
          yaml << "outputs:\n";
          yaml << "  - name: out\n";
          yaml << "    dims: [1]\n";
          yaml << "    data_type: float32\n";
          yaml << "batch_coalesce_timeout_ms: 1\n";
          yaml << "max_batch_size: 1\n";
          yaml << "pool_size: 1\n";
          return yaml.str();
        }(),
        std::nullopt},
    InvalidConfigCase{
        "MissingDimsSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name: config_loader_test\n";
          yaml += "model: {{MODEL_PATH}}\n";
          yaml += "inputs:\n";
          yaml += "  - name: in\n";
          yaml += "    data_type: float32\n";
          yaml += "outputs:\n";
          yaml += "  - name: out\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "batch_coalesce_timeout_ms: 1\n";
          yaml += "max_batch_size: 1\n";
          yaml += "pool_size: 1\n";
          return yaml;
        }(),
        std::nullopt},
    InvalidConfigCase{
        "MissingDataTypeSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name: config_loader_test\n";
          yaml += "model: {{MODEL_PATH}}\n";
          yaml += "inputs:\n";
          yaml += "  - name: in\n";
          yaml += "    dims: [1]\n";
          yaml += "outputs:\n";
          yaml += "  - name: out\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "batch_coalesce_timeout_ms: 1\n";
          yaml += "max_batch_size: 1\n";
          yaml += "pool_size: 1\n";
          return yaml;
        }(),
        std::nullopt},
    InvalidConfigCase{
        "InvalidVerbositySetsValidFalse",
        [] {
          auto yaml = base_model_yaml();
          yaml += "verbosity: unknown\n";
          return yaml;
        }(),
        std::nullopt, false, false},
    InvalidConfigCase{
        "NonScalarNameSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name:\n";
          yaml += "  nested: config_loader_test\n";
          yaml += "model: {{MODEL_PATH}}\n";
          yaml += "inputs:\n";
          yaml += "  - name: in\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "outputs:\n";
          yaml += "  - name: out\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "batch_coalesce_timeout_ms: 1\n";
          yaml += "max_batch_size: 1\n";
          yaml += "pool_size: 1\n";
          return yaml;
        }(),
        "Configuration option 'name' must be a scalar string"},
    InvalidConfigCase{
        "MissingNameSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "model: {{MODEL_PATH}}\n";
          yaml += "inputs:\n";
          yaml += "  - name: in\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "outputs:\n";
          yaml += "  - name: out\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "batch_coalesce_timeout_ms: 1\n";
          yaml += "max_batch_size: 1\n";
          yaml += "pool_size: 1\n";
          return yaml;
        }(),
        "Missing required key: name"},
    InvalidConfigCase{
        "MissingModelSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name: config_loader_test\n";
          yaml += "inputs:\n";
          yaml += "  - name: in\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "outputs:\n";
          yaml += "  - name: out\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "batch_coalesce_timeout_ms: 1\n";
          yaml += "max_batch_size: 1\n";
          yaml += "pool_size: 1\n";
          return yaml;
        }(),
        std::nullopt, false, false},
    InvalidConfigCase{
        "NonexistentModelFileSetsValidFalse", base_model_yaml(), std::nullopt,
        true, false},
    InvalidConfigCase{
        "MissingInputSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name: config_loader_test\n";
          yaml += "model: {{MODEL_PATH}}\n";
          yaml += "outputs:\n";
          yaml += "  - name: out\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "batch_coalesce_timeout_ms: 1\n";
          yaml += "max_batch_size: 1\n";
          yaml += "pool_size: 1\n";
          return yaml;
        }(),
        std::nullopt},
    InvalidConfigCase{
        "MissingOutputSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name: config_loader_test\n";
          yaml += "model: {{MODEL_PATH}}\n";
          yaml += "inputs:\n";
          yaml += "  - name: in\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "batch_coalesce_timeout_ms: 1\n";
          yaml += "max_batch_size: 1\n";
          yaml += "pool_size: 1\n";
          return yaml;
        }(),
        std::nullopt},
    InvalidConfigCase{
        "TooManyInputsSetsValidFalse",
        [] {
          std::ostringstream yaml;
          yaml << "name: config_loader_test\n";
          yaml << "model: {{MODEL_PATH}}\n";
          yaml << "inputs:\n";
          for (std::size_t i = 0; i <= kMaxInputs; ++i) {
            yaml << "  - name: in" << i << "\n";
            yaml << "    dims: [1]\n";
            yaml << "    data_type: float32\n";
          }
          yaml << "outputs:\n";
          yaml << "  - name: out\n";
          yaml << "    dims: [1]\n";
          yaml << "    data_type: float32\n";
          yaml << "batch_coalesce_timeout_ms: 1\n";
          yaml << "max_batch_size: 1\n";
          yaml << "pool_size: 1\n";
          return yaml.str();
        }(),
        std::nullopt},
    InvalidConfigCase{
        "TooManyDimsSetsValidFalse",
        [] {
          std::ostringstream yaml;
          yaml << "name: config_loader_test\n";
          yaml << "model: {{MODEL_PATH}}\n";
          yaml << "inputs:\n";
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
          yaml << "outputs:\n";
          yaml << "  - name: out\n";
          yaml << "    dims: [1]\n";
          yaml << "    data_type: float32\n";
          yaml << "batch_coalesce_timeout_ms: 1\n";
          yaml << "max_batch_size: 1\n";
          yaml << "pool_size: 1\n";
          return yaml.str();
        }(),
        std::nullopt},
};

}  // namespace

TEST_P(InvalidConfigTest, MarksConfigInvalid)
{
  const auto& test_case = GetParam();

  std::optional<std::filesystem::path> model_path;
  if (test_case.needs_model_path) {
    const auto model_name =
        std::string{"config_loader_model_"} + test_case.name + ".pt";
    if (test_case.create_model_file) {
      model_path = WriteTempFile(model_name, std::string(1, '\0'));
    } else {
      model_path = std::filesystem::temp_directory_path() / model_name;
      std::filesystem::remove(*model_path);
    }
  }

  std::string yaml = test_case.yaml;
  if (test_case.needs_model_path && model_path.has_value()) {
    const std::string placeholder = "{{MODEL_PATH}}";
    const std::string replacement = model_path->string();
    std::size_t pos = 0;
    while ((pos = yaml.find(placeholder, pos)) != std::string::npos) {
      yaml.replace(pos, placeholder.size(), replacement);
      pos += replacement.size();
    }
  }

  const auto config_path = WriteTempFile(
      std::string{"config_loader_invalid_"} + test_case.name + ".yaml", yaml);

  std::unique_ptr<starpu_server::CaptureStream> capture;
  if (test_case.expected_error.has_value()) {
    capture = std::make_unique<starpu_server::CaptureStream>(std::cerr);
  }

  const RuntimeConfig cfg = load_config(config_path.string());

  if (capture) {
    const std::string expected =
        expected_log_line(ErrorLevel, *test_case.expected_error);
    EXPECT_EQ(capture->str(), expected);
  }

  EXPECT_FALSE(cfg.valid);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidConfigs, InvalidConfigTest, ::testing::ValuesIn(kInvalidConfigCases),
    InvalidConfigCaseName);

TEST(ConfigLoader, AllowsBooleanUseCuda)
{
  const auto model_path = WriteTempFile(
      "config_loader_scalar_use_cuda_model.pt", std::string(1, '\0'));

  std::string yaml = base_model_yaml();
  const std::string placeholder = "{{MODEL_PATH}}";
  const std::string replacement = model_path.string();
  std::size_t pos = 0;
  while ((pos = yaml.find(placeholder, pos)) != std::string::npos) {
    yaml.replace(pos, placeholder.size(), replacement);
    pos += replacement.size();
  }
  yaml += "use_cuda: true\n";

  const auto config_path =
      WriteTempFile("config_loader_scalar_use_cuda.yaml", yaml);

  const RuntimeConfig cfg = load_config(config_path.string());
  EXPECT_TRUE(cfg.valid);
  EXPECT_TRUE(cfg.devices.use_cuda);
  EXPECT_TRUE(cfg.devices.ids.empty());
}

TEST(ConfigLoader, RejectsNonMappingRoot)
{
  const auto config_path =
      WriteTempFile("config_loader_non_mapping_root.yaml", "- item\n");
  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(config_path.string());
  const std::string expected =
      expected_log_line(ErrorLevel, "Config root must be a mapping");
  EXPECT_EQ(capture.str(), expected);
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, RejectsEmptyConfig)
{
  const auto config_path =
      WriteTempFile("config_loader_empty.yaml", std::string{});
  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(config_path.string());
  const std::string expected =
      expected_log_line(ErrorLevel, "Config root must be a mapping");
  EXPECT_EQ(capture.str(), expected);
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, LoadsStarpuEnvVariables)
{
  const auto model_path =
      WriteTempFile("config_loader_starpu_env_model.pt", std::string(1, '\0'));

  std::string yaml = base_model_yaml();
  const std::string placeholder = "{{MODEL_PATH}}";
  const std::string replacement = model_path.string();
  std::size_t pos = 0;
  while ((pos = yaml.find(placeholder, pos)) != std::string::npos) {
    yaml.replace(pos, placeholder.size(), replacement);
    pos += replacement.size();
  }
  yaml += "starpu_env:\n";
  yaml += "  VAR_ONE: VALUE1\n";
  yaml += "  VAR_TWO: VALUE2\n";

  const auto config_path = WriteTempFile("config_loader_starpu_env.yaml", yaml);

  const RuntimeConfig cfg = load_config(config_path.string());
  EXPECT_TRUE(cfg.valid);
  auto it_one = cfg.starpu_env.find("VAR_ONE");
  ASSERT_NE(it_one, cfg.starpu_env.end());
  EXPECT_EQ(it_one->second, "VALUE1");
  auto it_two = cfg.starpu_env.find("VAR_TWO");
  ASSERT_NE(it_two, cfg.starpu_env.end());
  EXPECT_EQ(it_two->second, "VALUE2");
}

TEST(ConfigLoader, LoadsValidConfig)
{
  const auto model_path =
      std::filesystem::temp_directory_path() / "config_loader_valid_model.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "name: fcfs_config\n";
  yaml << "scheduler: fcfs\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "use_cpu: true\n";
  yaml << "use_cuda:\n";
  yaml << "  - { device_ids: [0, 1] }\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1, 3, 224, 224]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1, 1000]\n";
  yaml << "    data_type: float32\n";
  yaml << "verbosity: 3\n";
  yaml << "max_batch_size: 4\n";
  yaml << "pool_size: 2\n";
  yaml << "dynamic_batching: true\n";
  yaml << "batch_coalesce_timeout_ms: 15\n";
  yaml << "pregen_inputs: 8\n";
  yaml << "warmup_pregen_inputs: 5\n";
  yaml << "warmup_request_nb: 3\n";
  yaml << "warmup_batches_per_worker: 2\n";
  yaml << "seed: 123\n";
  yaml << "validate_results: false\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_valid.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.name, "fcfs_config");
  EXPECT_EQ(cfg.scheduler, "fcfs");
  EXPECT_EQ(cfg.models[0].path, model_path.string());
  EXPECT_EQ(cfg.devices.ids, (std::vector<int>{0, 1}));
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
  EXPECT_EQ(cfg.batching.max_batch_size, 4);
  EXPECT_TRUE(cfg.batching.dynamic_batching);
  EXPECT_EQ(cfg.batching.batch_coalesce_timeout_ms, 15);
  EXPECT_EQ(cfg.batching.pregen_inputs, 8U);
  EXPECT_EQ(cfg.batching.warmup_pregen_inputs, 5U);
  EXPECT_EQ(cfg.batching.warmup_request_nb, 3);
  EXPECT_EQ(cfg.batching.warmup_batches_per_worker, 2);
  const bool has_seed = cfg.seed.has_value();
  ASSERT_TRUE(has_seed);
  const auto seed_value = cfg.seed.value_or(0U);
  EXPECT_EQ(seed_value, 123U);
  EXPECT_FALSE(cfg.validation.validate_results);
  EXPECT_TRUE(cfg.devices.use_cuda);
  EXPECT_FALSE(cfg.devices.group_cpu_by_numa);
}

TEST(ConfigLoader, NonSequenceInputYieldsEmptyTensorList)
{
  const auto model_path = std::filesystem::temp_directory_path() /
                          "config_loader_non_sequence_input_model.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "name: config_loader_test\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  name: bogus\n";
  yaml << "  dims: [1]\n";
  yaml << "  data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "batch_coalesce_timeout_ms: 1\n";
  yaml << "max_batch_size: 1\n";
  yaml << "pool_size: 1\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_non_sequence_input.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  ASSERT_EQ(cfg.models.size(), 1U);
  EXPECT_TRUE(cfg.models[0].inputs.empty());
  ASSERT_EQ(cfg.models[0].outputs.size(), 1U);
  EXPECT_EQ(cfg.models[0].outputs[0].name, "out");
  EXPECT_EQ(cfg.models[0].outputs[0].dims, (std::vector<int64_t>{1}));
  EXPECT_EQ(cfg.models[0].outputs[0].type, at::kFloat);
}

TEST(ConfigLoader, ParsesRuntimeFlags)
{
  const auto model_path =
      std::filesystem::temp_directory_path() / "config_loader_flags_model.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "name: runtime_flags\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "batch_coalesce_timeout_ms: 1\n";
  yaml << "max_batch_size: 1\n";
  yaml << "pool_size: 1\n";
  yaml << "rtol: 1.0e-4\n";
  yaml << "atol: 2.0e-5\n";
  yaml << "sync: true\n";
  yaml << "group_cpu_by_numa: true\n";
  yaml << "use_cpu: false\n";
  yaml << "use_cuda:\n";
  yaml << "  - { device_ids: [0] }\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_flags.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_DOUBLE_EQ(cfg.validation.rtol, 1.0e-4);
  EXPECT_DOUBLE_EQ(cfg.validation.atol, 2.0e-5);
  EXPECT_TRUE(cfg.batching.synchronous);
  EXPECT_TRUE(cfg.devices.group_cpu_by_numa);
  EXPECT_FALSE(cfg.devices.use_cpu);
  EXPECT_TRUE(cfg.devices.use_cuda);
}

TEST(ConfigLoader, ParsesVerboseAlias)
{
  const auto model_path =
      std::filesystem::temp_directory_path() / "config_loader_verbose_model.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "name: verbose_alias\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "verbose: trace\n";
  yaml << "batch_coalesce_timeout_ms: 1\n";
  yaml << "max_batch_size: 1\n";
  yaml << "pool_size: 1\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_verbose.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.verbosity, VerbosityLevel::Trace);
}

TEST(ConfigLoader, ParsesMaxMessageBytesAndInputSlots)
{
  const auto model_path =
      std::filesystem::temp_directory_path() / "config_loader_slots_model.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "name: slots_test\n";
  yaml << "scheduler: fcfs\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "max_message_bytes: 4096\n";
  yaml << "batch_coalesce_timeout_ms: 1\n";
  yaml << "max_batch_size: 1\n";
  yaml << "pool_size: 3\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_slots.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.batching.max_message_bytes, 4096U);
  EXPECT_EQ(cfg.batching.pool_size, 3);
}

TEST(ConfigLoader, ParsesTraceEnabledFlag)
{
  const auto model_path = WriteTempFile(
      "config_loader_trace_enabled_model.pt", std::string(1, '\0'));

  std::string yaml = base_model_yaml();
  const std::string placeholder = "{{MODEL_PATH}}";
  const std::string replacement = model_path.string();
  std::size_t pos = 0;
  while ((pos = yaml.find(placeholder, pos)) != std::string::npos) {
    yaml.replace(pos, placeholder.size(), replacement);
    pos += replacement.size();
  }
  yaml += "trace_enabled: true\n";

  const auto tmp = WriteTempFile("config_loader_trace_enabled.yaml", yaml);

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_TRUE(cfg.batching.trace_enabled);
}

TEST(ConfigLoader, ParsesTraceOutputDirectory)
{
  const auto model_path = WriteTempFile(
      "config_loader_trace_output_model.pt", std::string(1, '\0'));

  std::string yaml = base_model_yaml();
  const std::string placeholder = "{{MODEL_PATH}}";
  const std::string replacement = model_path.string();
  std::size_t pos = 0;
  while ((pos = yaml.find(placeholder, pos)) != std::string::npos) {
    yaml.replace(pos, placeholder.size(), replacement);
    pos += replacement.size();
  }

  const auto trace_dir =
      std::filesystem::temp_directory_path() / "config_loader_trace_dir";
  std::filesystem::create_directories(trace_dir);
  yaml += "trace_output: " + trace_dir.string() + "\n";

  const auto tmp = WriteTempFile("config_loader_trace_output.yaml", yaml);

  const RuntimeConfig cfg = load_config(tmp.string());

  const auto expected_path = (trace_dir / "batching_trace.json").string();
  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.batching.trace_output_path, expected_path);
}

TEST(ConfigLoader, TraceOutputRejectsEmptyPath)
{
  const auto model_path =
      WriteTempFile("config_loader_empty_trace_model.pt", std::string(1, '\0'));

  std::string yaml = base_model_yaml();
  const std::string placeholder = "{{MODEL_PATH}}";
  const std::string replacement = model_path.string();
  std::size_t pos = 0;
  while ((pos = yaml.find(placeholder, pos)) != std::string::npos) {
    yaml.replace(pos, placeholder.size(), replacement);
    pos += replacement.size();
  }
  yaml += "trace_output: \"\"\n";

  const auto tmp = WriteTempFile("config_loader_empty_trace.yaml", yaml);

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: trace_output must not be empty";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, TraceOutputAcceptsExistingDirectoryPath)
{
  const auto model_path =
      WriteTempFile("config_loader_trace_dir_model.pt", std::string(1, '\0'));

  std::string yaml = base_model_yaml();
  const std::string placeholder = "{{MODEL_PATH}}";
  const std::string replacement = model_path.string();
  std::size_t pos = 0;
  while ((pos = yaml.find(placeholder, pos)) != std::string::npos) {
    yaml.replace(pos, placeholder.size(), replacement);
    pos += replacement.size();
  }

  const auto trace_dir = MakeUniqueTempDir("config_loader_trace_dir");
  yaml += "trace_output: " + trace_dir.string() + "\n";

  const auto tmp = WriteTempFile("config_loader_trace_dir.yaml", yaml);

  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_path =
      (trace_dir / RuntimeConfig{}.batching.trace_output_path).string();
  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.batching.trace_output_path, expected_path);
}

TEST(ConfigLoader, TraceOutputAcceptsDirectoryWithTrailingSeparator)
{
  const auto model_path = WriteTempFile(
      "config_loader_trace_dir_sep_model.pt", std::string(1, '\0'));

  std::string yaml = base_model_yaml();
  const std::string placeholder = "{{MODEL_PATH}}";
  const std::string replacement = model_path.string();
  std::size_t pos = 0;
  while ((pos = yaml.find(placeholder, pos)) != std::string::npos) {
    yaml.replace(pos, placeholder.size(), replacement);
    pos += replacement.size();
  }

  const auto parent = MakeUniqueTempDir("config_loader_trace_dir_sep");
  const auto requested_dir = parent / "nested_dir";
  auto dir_with_separator = requested_dir.string();
  if (dir_with_separator.empty() ||
      dir_with_separator.back() != std::filesystem::path::preferred_separator) {
    dir_with_separator.push_back(std::filesystem::path::preferred_separator);
  }

  yaml += "trace_output: \"" + dir_with_separator + "\"\n";

  const auto tmp = WriteTempFile("config_loader_trace_dir_sep.yaml", yaml);

  const RuntimeConfig cfg = load_config(tmp.string());
  const std::string expected_path =
      (requested_dir / RuntimeConfig{}.batching.trace_output_path).string();

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.batching.trace_output_path, expected_path);
}

TEST(ConfigLoader, TraceOutputRejectsExplicitJsonFilename)
{
  const auto model_path =
      WriteTempFile("config_loader_trace_json_model.pt", std::string(1, '\0'));

  std::string yaml = base_model_yaml();
  const std::string placeholder = "{{MODEL_PATH}}";
  const std::string replacement = model_path.string();
  std::size_t pos = 0;
  while ((pos = yaml.find(placeholder, pos)) != std::string::npos) {
    yaml.replace(pos, placeholder.size(), replacement);
    pos += replacement.size();
  }

  const auto invalid_path =
      (std::filesystem::temp_directory_path() / "config_loader_trace.json")
          .string();
  yaml += "trace_output: " + invalid_path + "\n";

  const auto tmp = WriteTempFile("config_loader_trace_json.yaml", yaml);

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: trace_output must be a directory path (omit the "
      "filename)";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, TraceOutputRejectsPathPointingToExistingFile)
{
  const auto model_path = WriteTempFile(
      "config_loader_trace_file_path_model.pt", std::string(1, '\0'));

  std::string yaml = base_model_yaml();
  const std::string placeholder = "{{MODEL_PATH}}";
  const std::string replacement = model_path.string();
  std::size_t pos = 0;
  while ((pos = yaml.find(placeholder, pos)) != std::string::npos) {
    yaml.replace(pos, placeholder.size(), replacement);
    pos += replacement.size();
  }

  const auto destination =
      WriteTempFile("config_loader_trace_destination", "payload");
  yaml += "trace_output: " + destination.string() + "\n";

  const auto tmp = WriteTempFile("config_loader_trace_destination.yaml", yaml);

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: trace_output must be a directory path";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));
  EXPECT_FALSE(cfg.valid);
}

TEST(
    ConfigLoader, InvalidDimensionDuringMaxMessageComputationMarksConfigInvalid)
{
  const auto model_path =
      WriteTempFile("config_loader_invalid_dim_model.pt", std::string(1, '\0'));

  std::string yaml = base_model_yaml();
  const std::string placeholder = "{{MODEL_PATH}}";
  const std::string replacement = model_path.string();
  std::size_t pos = 0;
  while ((pos = yaml.find(placeholder, pos)) != std::string::npos) {
    yaml.replace(pos, placeholder.size(), replacement);
    pos += replacement.size();
  }

  const auto tmp = WriteTempFile("config_loader_invalid_dim.yaml", yaml);

  ConfigLoaderHookGuard hook_guard([](RuntimeConfig& cfg) {
    if (!cfg.models.empty() && !cfg.models[0].inputs.empty()) {
      cfg.models[0].inputs[0].dims[0] = -1;
    }
  });

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: dimension size must be non-negative";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, FilesystemErrorsMarkConfigInvalid)
{
  const auto protected_dir = MakeUniqueTempDir("config_loader_no_access");
  ScopedPermissionRestorer cleanup(protected_dir);

  const auto protected_model = protected_dir / "model.pt";
  std::ofstream(protected_model).put('\0');

  std::filesystem::permissions(
      protected_dir, std::filesystem::perms::none,
      std::filesystem::perm_options::replace);

  std::string expected_error_message;
  try {
    [[maybe_unused]] const bool exists_result =
        std::filesystem::exists(protected_model);
  }
  catch (const std::filesystem::filesystem_error& error) {
    expected_error_message =
        std::string{"Failed to load config: "} + error.what();
  }
  ASSERT_FALSE(expected_error_message.empty());

  std::string yaml = base_model_yaml();
  const std::string placeholder = "{{MODEL_PATH}}";
  const std::string replacement = protected_model.string();
  std::size_t pos = 0;
  while ((pos = yaml.find(placeholder, pos)) != std::string::npos) {
    yaml.replace(pos, placeholder.size(), replacement);
    pos += replacement.size();
  }

  const auto tmp = WriteTempFile("config_loader_filesystem_error.yaml", yaml);

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(
      capture.str(), expected_log_line(ErrorLevel, expected_error_message));
}

TEST(ConfigLoader, MaxMessageBytesRejectsNegative)
{
  const auto model_path = std::filesystem::temp_directory_path() /
                          "config_loader_negative_bytes_model.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "name: negative_bytes\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "max_message_bytes: -1\n";
  yaml << "batch_coalesce_timeout_ms: 1\n";
  yaml << "max_batch_size: 1\n";
  yaml << "pool_size: 1\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_negative_bytes.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: max_message_bytes must be >= 0 and fit in size_t";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, MaxBatchSizeRejectsNonPositive)
{
  const auto model_path = std::filesystem::temp_directory_path() /
                          "config_loader_zero_batch_model.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "name: negative_value\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "max_batch_size: 0\n";
  yaml << "batch_coalesce_timeout_ms: 1\n";
  yaml << "pool_size: 1\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_zero_batch.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: max_batch_size must be > 0";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

struct NegativeValueCase {
  const char* key;
  const char* value;
  const char* error;
};

class NegativeRuntimeValueCase
    : public ::testing::TestWithParam<NegativeValueCase> {};

TEST_P(NegativeRuntimeValueCase, MarksConfigInvalid)
{
  const auto [key, value, error] = GetParam();
  const auto model_path = std::filesystem::temp_directory_path() /
                          "config_loader_negative_value.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "name: zero_pool\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << key << ": " << value << "\n";
  yaml << "batch_coalesce_timeout_ms: 1\n";
  yaml << "max_batch_size: 1\n";
  yaml << "pool_size: 1\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_negative_value.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      std::string{"Failed to load config: "} + error;
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));
  EXPECT_FALSE(cfg.valid);
}

INSTANTIATE_TEST_SUITE_P(
    NegativeRuntimeValues, NegativeRuntimeValueCase,
    ::testing::Values(
        NegativeValueCase{"pregen_inputs", "0", "pregen_inputs must be > 0"},
        NegativeValueCase{
            "warmup_pregen_inputs", "0", "warmup_pregen_inputs must be > 0"},
        NegativeValueCase{
            "warmup_request_nb", "-1", "warmup_request_nb must be >= 0"},
        NegativeValueCase{
            "warmup_batches_per_worker", "-1",
            "warmup_batches_per_worker must be >= 0"},
        NegativeValueCase{"seed", "-1", "seed must be >= 0"},
        NegativeValueCase{"rtol", "-1.0", "rtol must be >= 0"},
        NegativeValueCase{"atol", "-1.0", "atol must be >= 0"}));

TEST(ConfigLoader, PoolSizeRejectsNonPositive)
{
  const auto model_path = std::filesystem::temp_directory_path() /
                          "config_loader_zero_slots_model.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "name: delay_addr\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "max_batch_size: 1\n";
  yaml << "batch_coalesce_timeout_ms: 1\n";
  yaml << "pool_size: 0\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_zero_slots.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: pool_size must be > 0";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, ParsesDelayAndAddress)
{
  const auto model_path = std::filesystem::temp_directory_path() /
                          "config_loader_delay_addr_model.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "name: overflow_dims\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "max_batch_size: 2\n";
  yaml << "batch_coalesce_timeout_ms: 1\n";
  yaml << "pool_size: 2\n";
  yaml << "delay_us: 15\n";
  yaml << "address: 127.0.0.1:50051\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_delay_addr.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.batching.delay_us, 15);
  EXPECT_EQ(cfg.server_address, "127.0.0.1:50051");
}

TEST(ConfigLoader, MissingModelSkipsParsingOtherKeys)
{
  const std::string yaml = R"(
name: config_loader_test
inputs:
  - name: in
    dims: [1]
    data_type: float32
outputs:
  - name: out
    dims: [1]
    data_type: float32
max_batch_size: 0
batch_coalesce_timeout_ms: 1
pool_size: 1
)";
  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_no_model_skip.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(cfg.batching.max_batch_size, 1);
}

TEST(ConfigLoader, MissingInputSkipsParsingOtherKeys)
{
  const std::string yaml = R"(
name: config_loader_test
model: model.pt
outputs:
  - name: out
    dims: [1]
    data_type: float32
delay_us: -10
max_batch_size: 1
batch_coalesce_timeout_ms: 1
pool_size: 1
)";
  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_no_input_skip.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(cfg.batching.delay_us, 0);
}

TEST(ConfigLoader, MissingOutputSkipsParsingOtherKeys)
{
  const std::string yaml = R"(
name: config_loader_test
model: model.pt
inputs:
  - name: in
    dims: [1]
    data_type: float32
max_batch_size: 0
batch_coalesce_timeout_ms: 1
pool_size: 1
)";
  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_no_output_skip.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(cfg.batching.max_batch_size, 1);
}

TEST(
    ConfigLoader,
    MessageSizeOverflowDuringMaxMessageComputationMarksConfigInvalid)
{
  const auto model_path = std::filesystem::temp_directory_path() /
                          "config_loader_overflow_dims_model.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "name: overflow_dims\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: huge\n";
  yaml << "    dims: [2147483647, 2147483647, 2147483647]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "batch_coalesce_timeout_ms: 1\n";
  yaml << "max_batch_size: 1\n";
  yaml << "pool_size: 1\n";

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

  // InvalidDimensionException cannot be triggered via YAML alone because
  // parse_tensor_nodes already rejects non-positive dimensions when reading the
  // config. A dedicated test uses the config loader post-parse hook to mutate
  // the parsed model and cover that catch.
}

TEST(ConfigLoader, UnsupportedDtypeDuringMaxMessageComputationMarksInvalid)
{
  const auto model_path = std::filesystem::temp_directory_path() /
                          "config_loader_complex_dtype_model.pt";
  std::ofstream(model_path).put('\0');

  std::ostringstream yaml;
  yaml << "name: complex_dtype\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: complex_input\n";
  yaml << "    dims: [1, 1]\n";
  yaml << "    data_type: complex64\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "batch_coalesce_timeout_ms: 1\n";
  yaml << "max_batch_size: 1\n";
  yaml << "pool_size: 1\n";

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
  const auto model_path = WriteTempFile(
      std::format("config_loader_verbosity_{}.pt", value),
      std::string(1, '\0'));

  std::string yaml = base_model_yaml();
  const std::string placeholder = "{{MODEL_PATH}}";
  const std::string replacement = model_path.string();
  std::size_t pos = 0;
  while ((pos = yaml.find(placeholder, pos)) != std::string::npos) {
    yaml.replace(pos, placeholder.size(), replacement);
    pos += replacement.size();
  }
  yaml += std::format("verbosity: {}\n", value);

  const auto tmp = WriteTempFile(
      std::format("config_loader_verbosity_{}.yaml", value), yaml);

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
