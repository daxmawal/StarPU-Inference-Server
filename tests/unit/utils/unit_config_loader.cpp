#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
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
NextTempArtifactSequence() -> std::uint64_t
{
  static std::atomic<std::uint64_t> sequence{0};
  return sequence.fetch_add(1, std::memory_order_relaxed);
}

auto
DeterministicTempPath(const std::string& name) -> std::filesystem::path
{
  const auto sequence = NextTempArtifactSequence();
  return std::filesystem::temp_directory_path() /
         (std::to_string(sequence) + "_" + name);
}

auto
WriteTempFile(const std::string& name, const std::string& contents)
    -> std::filesystem::path
{
  const auto path = DeterministicTempPath(name);
  std::ofstream(path) << contents;
  return path;
}

auto
WriteEmptyModelFile(const std::string& name) -> std::filesystem::path
{
  const auto path = DeterministicTempPath(name);
  std::ofstream(path).put('\0');
  return path;
}

auto
ReplaceModelPath(std::string yaml, const std::filesystem::path& model_path)
    -> std::string
{
  const std::string placeholder = "{{MODEL_PATH}}";
  const std::string replacement = model_path.string();
  std::size_t pos = 0;
  while ((pos = yaml.find(placeholder, pos)) != std::string::npos) {
    yaml.replace(pos, placeholder.size(), replacement);
    pos += replacement.size();
  }
  return yaml;
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
  const auto unique_suffix = NextTempArtifactSequence();
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
        "Failed to load config: Unknown configuration option: unknown_option"},
    InvalidConfigCase{
        "NonScalarKeySetsValidFalse",
        [] {
          auto yaml = base_model_yaml();
          yaml += "? [invalid, key]\n";
          yaml += ": true\n";
          return yaml;
        }(),
        "Failed to load config: Configuration keys must be scalar strings"},
    InvalidConfigCase{
        "DeviceIdsAtRootInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "device_ids: [0]\n";
          return yaml;
        }(),
        "Failed to load config: device_ids must be nested inside the use_cuda "
        "block (e.g. \"use_cuda: [{ device_ids: [0] }]\")"},
    InvalidConfigCase{
        "InvalidConfigSetsValidFalse",
        [] {
          auto yaml = base_model_yaml();
          yaml += "max_batch_size: 0\n";
          return yaml;
        }(),
        std::nullopt, false, false},
    InvalidConfigCase{
        "InvalidYamlSyntaxSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name: [1, 2\n";
          return yaml;
        }(),
        std::nullopt, false, false},
    InvalidConfigCase{
        "UseCudaEmptySequenceInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "use_cuda: []\n";
          return yaml;
        }(),
        "Failed to load config: use_cuda requires at least one device_ids "
        "entry"},
    InvalidConfigCase{
        "UseCudaNonSequenceInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "use_cuda:\n";
          yaml += "  device_ids: [0]\n";
          return yaml;
        }(),
        "Failed to load config: use_cuda must be a boolean or a sequence of "
        "device mappings"},
    InvalidConfigCase{
        "UseCudaEntryNotMapInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "use_cuda:\n";
          yaml += "  - true\n";
          yaml += "  - { device_ids: [0] }\n";
          return yaml;
        }(),
        "Failed to load config: use_cuda[0] must be a mapping that defines "
        "device_ids"},
    InvalidConfigCase{
        "UseCudaEntryMissingDeviceIdsInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "use_cuda:\n";
          yaml += "  - {}\n";
          yaml += "  - { device_ids: [0] }\n";
          return yaml;
        }(),
        "Failed to load config: use_cuda[0].device_ids is required"},
    InvalidConfigCase{
        "UseCudaEntryDeviceIdsNotSequenceInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "use_cuda:\n";
          yaml += "  - { device_ids: 0 }\n";
          yaml += "  - { device_ids: [1] }\n";
          return yaml;
        }(),
        "Failed to load config: use_cuda[0].device_ids must be a sequence of "
        "integers"},
    InvalidConfigCase{
        "BothBackendsDisabledInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "use_cpu: false\n";
          yaml += "use_cuda: false\n";
          return yaml;
        }(),
        "Failed to load config: At least one execution backend must be "
        "enabled: "
        "set use_cpu: true and/or configure use_cuda with device_ids"},
    InvalidConfigCase{
        "UseCudaNegativeDeviceIdInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "use_cuda:\n";
          yaml += "  - { device_ids: [-1] }\n";
          return yaml;
        }(),
        "Failed to load config: use_cuda[0].device_ids[0] must be >= 0"},
    InvalidConfigCase{
        "UseCudaDuplicatedDeviceIdInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "use_cuda:\n";
          yaml += "  - { device_ids: [0, 0] }\n";
          return yaml;
        }(),
        "Failed to load config: use_cuda[0].device_ids[1] is duplicated"},
    InvalidConfigCase{
        "UseCudaEmptyDeviceIdsInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "use_cuda:\n";
          yaml += "  - { device_ids: [] }\n";
          return yaml;
        }(),
        "Failed to load config: use_cuda requires at least one device_ids "
        "entry"},
    InvalidConfigCase{
        "StarpuEnvNotMapInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "starpu_env: []\n";
          return yaml;
        }(),
        "Failed to load config: starpu_env must be a mapping of variable "
        "names to values"},
    InvalidConfigCase{
        "StarpuEnvKeyNotScalarInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "starpu_env:\n";
          yaml += "  ? [invalid, key]\n";
          yaml += "  : value\n";
          return yaml;
        }(),
        "Failed to load config: starpu_env entries must have scalar keys"},
    InvalidConfigCase{
        "StarpuEnvValueNotScalarInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "starpu_env:\n";
          yaml += "  VAR: [1, 2]\n";
          return yaml;
        }(),
        "Failed to load config: starpu_env entry 'VAR' must have a scalar "
        "value"},
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
        "MetricsPortOutOfRangeSetsValidFalse",
        [] {
          auto yaml = base_model_yaml();
          yaml += "metrics_port: 70000\n";
          return yaml;
        }(),
        std::nullopt},
    InvalidConfigCase{
        "MetricsPortNotScalarInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "metrics_port: [1]\n";
          return yaml;
        }(),
        "Failed to load config: metrics_port must be an integer"},
    InvalidConfigCase{
        "MetricsPortBadConversionInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "metrics_port: not_a_number\n";
          return yaml;
        }(),
        "Failed to load config: metrics_port must be an integer"},
    InvalidConfigCase{
        "DeprecatedSchedulerOptionSetsValidFalse",
        [] {
          auto yaml = std::string{"scheduler: unknown\n"};
          yaml += base_model_yaml();
          return yaml;
        }(),
        "Failed to load config: Unknown configuration option: scheduler (use "
        "starpu_env with STARPU_SCHED)"},
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
        "Failed to load config: Configuration option 'name' must be a scalar "
        "string"},
    InvalidConfigCase{
        "NonScalarModelNameSetsValidFalse",
        [] {
          auto yaml = base_model_yaml();
          yaml += "model_name: [invalid, name]\n";
          return yaml;
        }(),
        "Failed to load config: model_name must be a scalar string"},
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
        "Failed to load config: Missing required key: name"},
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
        "EmptyModelPathSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name: config_loader_test\n";
          yaml += "model: \"\"\n";
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
        "Failed to load config: model must not be empty", false, false},
    InvalidConfigCase{
        "MissingMultipleRequiredKeysSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name: config_loader_test\n";
          return yaml;
        }(),
        "Failed to load config: Missing required keys: model, inputs, outputs, "
        "pool_size, max_batch_size, batch_coalesce_timeout_ms",
        false, false},
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
        "InputNameNotScalarSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name: config_loader_test\n";
          yaml += "model: {{MODEL_PATH}}\n";
          yaml += "inputs:\n";
          yaml += "  - name: [invalid]\n";
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
        "Failed to load config: inputs[0].name must be a scalar string"},
    InvalidConfigCase{
        "InputDimsNotSequenceSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name: config_loader_test\n";
          yaml += "model: {{MODEL_PATH}}\n";
          yaml += "inputs:\n";
          yaml += "  - name: in\n";
          yaml += "    dims: 1\n";
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
        "Failed to load config: inputs[0].dims must be a sequence of integers"},
    InvalidConfigCase{
        "InputDataTypeNotScalarSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name: config_loader_test\n";
          yaml += "model: {{MODEL_PATH}}\n";
          yaml += "inputs:\n";
          yaml += "  - name: in\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: [float32]\n";
          yaml += "outputs:\n";
          yaml += "  - name: out\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "batch_coalesce_timeout_ms: 1\n";
          yaml += "max_batch_size: 1\n";
          yaml += "pool_size: 1\n";
          return yaml;
        }(),
        "Failed to load config: inputs[0].data_type must be a scalar string"},
    InvalidConfigCase{
        "InputDataTypeInvalidSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name: config_loader_test\n";
          yaml += "model: {{MODEL_PATH}}\n";
          yaml += "inputs:\n";
          yaml += "  - name: in\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: not_a_dtype\n";
          yaml += "outputs:\n";
          yaml += "  - name: out\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "batch_coalesce_timeout_ms: 1\n";
          yaml += "max_batch_size: 1\n";
          yaml += "pool_size: 1\n";
          return yaml;
        }(),
        "Failed to load config: inputs[0].data_type: Unsupported type: "
        "not_a_dtype"},
    InvalidConfigCase{
        "InputsEntryNotMapSetsValidFalse",
        [] {
          std::string yaml;
          yaml += "name: config_loader_test\n";
          yaml += "model: {{MODEL_PATH}}\n";
          yaml += "inputs:\n";
          yaml += "  - in\n";
          yaml += "outputs:\n";
          yaml += "  - name: out\n";
          yaml += "    dims: [1]\n";
          yaml += "    data_type: float32\n";
          yaml += "batch_coalesce_timeout_ms: 1\n";
          yaml += "max_batch_size: 1\n";
          yaml += "pool_size: 1\n";
          return yaml;
        }(),
        "Failed to load config: inputs[0] must be a mapping"},
    InvalidConfigCase{
        "TooManyInputsSetsValidFalse",
        [] {
          std::ostringstream yaml;
          yaml << "name: config_loader_test\n";
          yaml << "model: {{MODEL_PATH}}\n";
          yaml << "inputs:\n";
          for (std::size_t i = 0; i <= InferLimits::MaxInputs; ++i) {
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
          for (std::size_t i = 0; i <= InferLimits::MaxDims; ++i) {
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
      model_path = WriteEmptyModelFile(model_name);
    } else {
      model_path = std::filesystem::temp_directory_path() / model_name;
      std::filesystem::remove(*model_path);
    }
  }

  std::string yaml = test_case.yaml;
  if (test_case.needs_model_path && model_path.has_value()) {
    yaml = ReplaceModelPath(std::move(yaml), *model_path);
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

TEST(ConfigLoader, ParseTensorNodesReturnsEmptyWhenUndefined)
{
  YAML::Node undefined(YAML::NodeType::Undefined);
  const auto tensors = parse_tensor_nodes_for_test(undefined, 4U, "inputs", 4U);
  EXPECT_TRUE(tensors.empty());
}

TEST(ConfigLoader, AcceptsBooleanUseCudaFalse)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_scalar_use_cuda_false_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "use_cuda: false\n";

  const auto config_path =
      WriteTempFile("config_loader_scalar_use_cuda_false.yaml", yaml);

  const RuntimeConfig cfg = load_config(config_path.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_FALSE(cfg.devices.use_cuda);
  EXPECT_TRUE(cfg.devices.ids.empty());
}

TEST(ConfigLoader, RejectsBooleanUseCudaTrue)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_scalar_use_cuda_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "use_cuda: true\n";

  const auto config_path =
      WriteTempFile("config_loader_scalar_use_cuda.yaml", yaml);

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(config_path.string());
  const std::string expected_error =
      "Failed to load config: use_cuda must be a sequence of device mappings "
      "when enabled (e.g. \"use_cuda: [{ device_ids: [0] }]\")";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, RejectsNonMappingRoot)
{
  const auto config_path =
      WriteTempFile("config_loader_non_mapping_root.yaml", "- item\n");
  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(config_path.string());
  const std::string expected = expected_log_line(
      ErrorLevel, "Failed to load config: Config root must be a mapping");
  EXPECT_EQ(capture.str(), expected);
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, RejectsEmptyConfig)
{
  const auto config_path =
      WriteTempFile("config_loader_empty.yaml", std::string{});
  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(config_path.string());
  const std::string expected = expected_log_line(
      ErrorLevel, "Failed to load config: Config root must be a mapping");
  EXPECT_EQ(capture.str(), expected);
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, LoadsStarpuEnvVariables)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_starpu_env_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
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
  const auto model_path = WriteEmptyModelFile("config_loader_valid_model.pt");

  std::ostringstream yaml;
  yaml << "name: fcfs_config\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "model_name: bert_model\n";
  yaml << "use_cpu: true\n";
  yaml << "use_cuda:\n";
  yaml << "  - { device_ids: [0, 1] }\n";
  yaml << "starpu_env:\n";
  yaml << "  STARPU_SCHED: pheft\n";
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
  yaml << "warmup_pregen_inputs: 5\n";
  yaml << "warmup_request_nb: 3\n";
  yaml << "warmup_batches_per_worker: 2\n";
  yaml << "seed: 123\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_valid.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  ASSERT_TRUE(cfg.model.has_value());
  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.name, "fcfs_config");
  EXPECT_EQ(cfg.model->path, model_path.string());
  EXPECT_EQ(cfg.model->name, "bert_model");
  EXPECT_EQ(cfg.devices.ids, (std::vector<int>{0, 1}));
  auto scheduler_env = cfg.starpu_env.find("STARPU_SCHED");
  ASSERT_NE(scheduler_env, cfg.starpu_env.end());
  EXPECT_EQ(scheduler_env->second, "pheft");
  ASSERT_EQ(cfg.model->inputs.size(), 1U);
  EXPECT_EQ(cfg.model->inputs[0].name, "in");
  EXPECT_EQ(cfg.model->inputs[0].dims, (std::vector<int64_t>{1, 3, 224, 224}));
  EXPECT_EQ(cfg.model->inputs[0].type, at::kFloat);
  ASSERT_EQ(cfg.model->outputs.size(), 1U);
  EXPECT_EQ(cfg.model->outputs[0].name, "out");
  EXPECT_EQ(cfg.model->outputs[0].dims, (std::vector<int64_t>{1, 1000}));
  EXPECT_EQ(cfg.model->outputs[0].type, at::kFloat);
  EXPECT_EQ(cfg.verbosity, VerbosityLevel::Debug);
  EXPECT_EQ(cfg.batching.max_batch_size, 4);
  EXPECT_TRUE(cfg.batching.dynamic_batching);
  EXPECT_EQ(cfg.batching.batch_coalesce_timeout_ms, 15);
  EXPECT_EQ(cfg.batching.warmup_pregen_inputs, 5U);
  EXPECT_EQ(cfg.batching.warmup_request_nb, 3);
  EXPECT_EQ(cfg.batching.warmup_batches_per_worker, 2);
  const bool has_seed = cfg.seed.has_value();
  ASSERT_TRUE(has_seed);
  const auto seed_value = cfg.seed.value_or(0U);
  EXPECT_EQ(seed_value, 123U);
  EXPECT_TRUE(cfg.devices.use_cuda);
  EXPECT_FALSE(cfg.devices.group_cpu_by_numa);
}

TEST(ConfigLoader, RejectsNonSequenceInput)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_non_sequence_input_model.pt");

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

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: inputs must be a sequence of tensor definitions";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, ParsesRuntimeFlags)
{
  const auto model_path = WriteEmptyModelFile("config_loader_flags_model.pt");

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
  EXPECT_TRUE(cfg.batching.synchronous);
  EXPECT_TRUE(cfg.devices.group_cpu_by_numa);
  EXPECT_FALSE(cfg.devices.use_cpu);
  EXPECT_TRUE(cfg.devices.use_cuda);
}

TEST(ConfigLoader, ParsesVerboseAlias)
{
  const auto model_path = WriteEmptyModelFile("config_loader_verbose_model.pt");

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
  const auto model_path = WriteEmptyModelFile("config_loader_slots_model.pt");

  std::ostringstream yaml;
  yaml << "name: slots_test\n";
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
  const auto model_path =
      WriteEmptyModelFile("config_loader_trace_enabled_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "trace_enabled: true\n";

  const auto tmp = WriteTempFile("config_loader_trace_enabled.yaml", yaml);

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_TRUE(cfg.batching.trace_enabled);
}

TEST(ConfigLoader, ParsesTraceOutputDirectory)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_trace_output_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);

  const auto trace_dir =
      std::filesystem::temp_directory_path() / "config_loader_trace_dir";
  std::filesystem::create_directories(trace_dir);
  yaml += "trace_output: " + trace_dir.string() + "\n";

  const auto tmp = WriteTempFile("config_loader_trace_output.yaml", yaml);

  const RuntimeConfig cfg = load_config(tmp.string());

  const auto expected_path =
      (trace_dir / std::string(kDefaultTraceFileName)).string();
  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.batching.trace_output_path, expected_path);
}

TEST(ConfigLoader, TraceOutputRejectsEmptyPath)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_empty_trace_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
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
      WriteEmptyModelFile("config_loader_trace_dir_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);

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
  const auto model_path =
      WriteEmptyModelFile("config_loader_trace_dir_sep_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);

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
      WriteEmptyModelFile("config_loader_trace_json_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);

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
  const auto model_path =
      WriteEmptyModelFile("config_loader_trace_file_path_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);

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

TEST(ConfigLoader, TraceOutputFilesystemErrorMarksConfigInvalid)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_trace_error_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);

  const auto protected_dir = MakeUniqueTempDir("config_loader_trace_no_access");
  ScopedPermissionRestorer cleanup(protected_dir);

  std::filesystem::permissions(
      protected_dir, std::filesystem::perms::none,
      std::filesystem::perm_options::replace);

  const auto protected_child = protected_dir / "trace_output";
  std::error_code status_ec;
  [[maybe_unused]] const bool exists_result =
      std::filesystem::exists(protected_child, status_ec);
  ASSERT_TRUE(status_ec);

  const std::filesystem::filesystem_error fs_error(
      "trace_output", protected_child, status_ec);
  const std::string expected_error =
      std::string{"Failed to load config: "} + fs_error.what();

  yaml += "trace_output: " + protected_child.string() + "\n";

  const auto tmp = WriteTempFile("config_loader_trace_error.yaml", yaml);

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));
}

TEST(
    ConfigLoader, InvalidDimensionDuringMaxMessageComputationMarksConfigInvalid)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_invalid_dim_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);

  const auto tmp = WriteTempFile("config_loader_invalid_dim.yaml", yaml);

  ConfigLoaderHookGuard hook_guard([](RuntimeConfig& cfg) {
    if (cfg.model.has_value() && !cfg.model->inputs.empty()) {
      cfg.model->inputs[0].dims[0] = -1;
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
  yaml = ReplaceModelPath(std::move(yaml), protected_model);

  const auto tmp = WriteTempFile("config_loader_filesystem_error.yaml", yaml);

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(
      capture.str(), expected_log_line(ErrorLevel, expected_error_message));
}

TEST(ConfigLoader, RejectsModelPathThatIsDirectory)
{
  const auto model_dir = MakeUniqueTempDir("config_loader_model_directory");
  ScopedPermissionRestorer cleanup(model_dir);

  std::string yaml = base_model_yaml();
  yaml = ReplaceModelPath(std::move(yaml), model_dir);

  const auto tmp = WriteTempFile("config_loader_model_directory.yaml", yaml);

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: Model path must be a regular file: " +
      model_dir.string();
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, RejectsUnreadableModelPath)
{
  const auto model_dir = MakeUniqueTempDir("config_loader_model_unreadable");
  ScopedPermissionRestorer cleanup(model_dir);

  const auto model_path = model_dir / "model.pt";
  std::ofstream(model_path).put('\0');
  std::filesystem::permissions(
      model_path, std::filesystem::perms::owner_write,
      std::filesystem::perm_options::replace);

  std::ifstream probe(model_path, std::ios::binary);
  if (probe.good()) {
    GTEST_SKIP() << "Model file remains readable under current privileges";
  }

  std::string yaml = base_model_yaml();
  yaml = ReplaceModelPath(std::move(yaml), model_path);
  const auto tmp = WriteTempFile("config_loader_model_unreadable.yaml", yaml);

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: Model path is not readable: " +
      model_path.string();
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, MaxMessageBytesRejectsNegative)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_negative_bytes_model.pt");

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

TEST(ConfigLoader, MaxMessageBytesRejectsTooSmallForModel)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_small_cap_model.pt");

  std::ostringstream yaml;
  yaml << "name: small_cap\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1, 4]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1, 4]\n";
  yaml << "    data_type: float32\n";
  yaml << "max_message_bytes: 16\n";
  yaml << "batch_coalesce_timeout_ms: 1\n";
  yaml << "max_batch_size: 1\n";
  yaml << "pool_size: 1\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_small_cap.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: max_message_bytes (16) is too small for "
      "configured model (requires at least 32 bytes)";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, MaxMessageBytesWarnsWhenExceedingGrpcLimit)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_grpc_limit_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  const auto grpc_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  const auto max_message_bytes = grpc_limit + 1U;
  yaml += std::format("max_message_bytes: {}\n", max_message_bytes);

  const auto tmp = WriteTempFile("config_loader_grpc_limit.yaml", yaml);

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  const std::string expected_warning = std::format(
      "max_message_bytes ({}) exceeds gRPC limit ({}); gRPC will clamp "
      "to {}. Consider reducing max_message_bytes.",
      max_message_bytes, grpc_limit, grpc_limit);
  EXPECT_EQ(capture.str(), expected_log_line(WarningLevel, expected_warning));
}

TEST(ConfigLoader, MaxBatchSizeRejectsNonPositive)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_zero_batch_model.pt");

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
  const auto model_path =
      WriteEmptyModelFile("config_loader_negative_value.pt");

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
        NegativeValueCase{
            "warmup_pregen_inputs", "-1", "warmup_pregen_inputs must be >= 0"},
        NegativeValueCase{
            "warmup_request_nb", "-1", "warmup_request_nb must be >= 0"},
        NegativeValueCase{
            "warmup_batches_per_worker", "-1",
            "warmup_batches_per_worker must be >= 0"},
        NegativeValueCase{"seed", "-1", "seed must be >= 0"}));

TEST(ConfigLoader, PoolSizeRejectsNonPositive)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_zero_slots_model.pt");

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

TEST(ConfigLoader, ParsesAddress)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_delay_addr_model.pt");

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
  yaml << "address: 127.0.0.1:50051\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_delay_addr.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.server_address, "127.0.0.1:50051");
}

TEST(ConfigLoader, ParsesMaxInflightTasks)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_max_inflight_tasks_model.pt");

  std::ostringstream yaml;
  yaml << "name: max_inflight_tasks_test\n";
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
  yaml << "pool_size: 1\n";
  yaml << "max_inflight_tasks: 100\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_max_inflight_tasks.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.batching.max_inflight_tasks, 100U);
}

TEST(ConfigLoader, MaxInflightTasksRejectsNegative)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_negative_inflight_model.pt");

  std::ostringstream yaml;
  yaml << "name: negative_inflight\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "max_inflight_tasks: -1\n";
  yaml << "batch_coalesce_timeout_ms: 1\n";
  yaml << "max_batch_size: 1\n";
  yaml << "pool_size: 1\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_negative_inflight.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: max_inflight_tasks must be >= 0 and fit in "
      "size_t";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, ParsesMaxQueueSize)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_max_queue_size_model.pt");

  std::ostringstream yaml;
  yaml << "name: max_queue_size_test\n";
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
  yaml << "pool_size: 1\n";
  yaml << "max_queue_size: 50\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_max_queue_size.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.batching.max_queue_size, 50U);
}

TEST(ConfigLoader, ParsesCongestionBlock)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_congestion_model.pt");

  std::ostringstream yaml;
  yaml << "name: congestion_config\n";
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
  yaml << "pool_size: 1\n";
  yaml << "congestion:\n";
  yaml << "  enabled: true\n";
  yaml << "  latency_slo_ms: 150\n";
  yaml << "  queue_latency_budget_ms: 30\n";
  yaml << "  queue_latency_budget_ratio: 0.25\n";
  yaml << "  e2e_warn_ratio: 0.9\n";
  yaml << "  e2e_ok_ratio: 0.8\n";
  yaml << "  fill_high: 0.85\n";
  yaml << "  fill_low: 0.65\n";
  yaml << "  rho_high: 1.1\n";
  yaml << "  rho_low: 0.9\n";
  yaml << "  alpha_ewma: 0.4\n";
  yaml << "  entry_horizon_ms: 3000\n";
  yaml << "  exit_horizon_ms: 7000\n";
  yaml << "  tick_interval_ms: 500\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_congestion.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  ASSERT_TRUE(cfg.valid);
  EXPECT_TRUE(cfg.congestion.enabled);
  EXPECT_DOUBLE_EQ(cfg.congestion.latency_slo_ms, 150.0);
  EXPECT_DOUBLE_EQ(cfg.congestion.queue_latency_budget_ms, 30.0);
  EXPECT_DOUBLE_EQ(cfg.congestion.queue_latency_budget_ratio, 0.25);
  EXPECT_DOUBLE_EQ(cfg.congestion.e2e_warn_ratio, 0.9);
  EXPECT_DOUBLE_EQ(cfg.congestion.e2e_ok_ratio, 0.8);
  EXPECT_DOUBLE_EQ(cfg.congestion.fill_high, 0.85);
  EXPECT_DOUBLE_EQ(cfg.congestion.fill_low, 0.65);
  EXPECT_DOUBLE_EQ(cfg.congestion.rho_high, 1.1);
  EXPECT_DOUBLE_EQ(cfg.congestion.rho_low, 0.9);
  EXPECT_DOUBLE_EQ(cfg.congestion.alpha, 0.4);
  EXPECT_EQ(cfg.congestion.entry_horizon_ms, 3000);
  EXPECT_EQ(cfg.congestion.exit_horizon_ms, 7000);
  EXPECT_EQ(cfg.congestion.tick_interval_ms, 500);
}

TEST(ConfigLoader, CongestionRejectsNonMapping)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_congestion_nonmap_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion: [invalid]\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_congestion_nonmap.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: congestion must be a mapping";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, CongestionQueueLatencyBudgetRatioRejectsNegative)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_negative_congestion_ratio_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += "  queue_latency_budget_ratio: -0.25\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_negative_congestion_ratio.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: queue_latency_budget_ratio must be >= 0";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, CongestionE2EWarnRatioRejectsNonPositive)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_negative_e2e_warn_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += "  e2e_warn_ratio: 0\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_negative_e2e_warn_ratio.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: e2e_warn_ratio must be > 0";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, CongestionE2EOkRatioRejectsNonPositive)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_negative_e2e_ok_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += "  e2e_warn_ratio: 0.9\n";
  yaml += "  e2e_ok_ratio: 0.0\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_negative_e2e_ok_ratio.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: e2e_ok_ratio must be > 0 and <= e2e_warn_ratio";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, CongestionE2EOkRatioRejectsAboveWarn)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_e2e_ok_above_warn_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += "  e2e_warn_ratio: 0.5\n";
  yaml += "  e2e_ok_ratio: 0.6\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_e2e_ok_above_warn.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: e2e_ok_ratio must be > 0 and <= e2e_warn_ratio";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, CongestionFillRejectsOutOfRange)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_invalid_fill_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += "  fill_high: 0\n";
  yaml += "  fill_low: 0.2\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_invalid_fill.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: fill_high must be (0,1] and fill_low in [0,1)";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, CongestionFillRejectsLowAboveHigh)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_fill_low_above_high_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += "  fill_high: 0.7\n";
  yaml += "  fill_low: 0.7\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_fill_low_above_high.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: fill_low must be < fill_high";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, CongestionRhoRejectsNonPositiveHigh)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_rho_high_nonpositive_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += "  rho_high: 0\n";
  yaml += "  rho_low: 0.5\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_rho_high_nonpositive.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: rho_high must be > 0";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, CongestionRhoRejectsNegativeLow)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_rho_low_negative_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += "  rho_high: 1.2\n";
  yaml += "  rho_low: -0.1\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_rho_low_negative.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: rho_low must be >= 0";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, CongestionRhoRejectsLowAboveHigh)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_rho_low_above_high_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += "  rho_high: 1.1\n";
  yaml += "  rho_low: 1.1\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_rho_low_above_high.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: rho_low must be < rho_high";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, CongestionEwmaRejectsNonPositive)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_alpha_nonpositive_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += "  alpha_ewma: 0\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_alpha_nonpositive.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: alpha_ewma must be in (0, 1]";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, CongestionEwmaRejectsAboveOne)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_alpha_above_one_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += "  alpha_ewma: 1.1\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_alpha_above_one.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: alpha_ewma must be in (0, 1]";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, CongestionHorizonsRejectNonPositiveEntry)
{
  YAML::Node congestion_node(YAML::NodeType::Map);
  RuntimeConfig cfg;
  cfg.congestion.entry_horizon_ms = 0;
  cfg.congestion.exit_horizon_ms = 100;

  try {
    parse_congestion_horizons_for_test(congestion_node, cfg);
    FAIL() << "Expected entry/exit horizon validation to fail.";
  }
  catch (const std::invalid_argument& error) {
    EXPECT_EQ(
        error.what(), std::string("entry_horizon_ms and exit_horizon_ms must "
                                  "be > 0"));
  }
}

TEST(ConfigLoader, CongestionHorizonsRejectNonPositiveExit)
{
  YAML::Node congestion_node(YAML::NodeType::Map);
  RuntimeConfig cfg;
  cfg.congestion.entry_horizon_ms = 100;
  cfg.congestion.exit_horizon_ms = 0;

  try {
    parse_congestion_horizons_for_test(congestion_node, cfg);
    FAIL() << "Expected entry/exit horizon validation to fail.";
  }
  catch (const std::invalid_argument& error) {
    EXPECT_EQ(
        error.what(), std::string("entry_horizon_ms and exit_horizon_ms must "
                                  "be > 0"));
  }
}

TEST(ConfigLoader, CongestionTickRejectsNonPositive)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_tick_nonpositive_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += "  tick_interval_ms: 0\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_tick_nonpositive.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: tick_interval_ms must be > 0";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, MaxQueueSizeRejectsNonPositive)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_zero_queue_model.pt");

  std::ostringstream yaml;
  yaml << "name: zero_queue\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "max_queue_size: 0\n";
  yaml << "batch_coalesce_timeout_ms: 1\n";
  yaml << "max_batch_size: 1\n";
  yaml << "pool_size: 1\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_zero_queue.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: max_queue_size must be > 0 and fit in size_t";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, BatchingCoherenceRejectsQueueLowerThanMaxBatch)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_incoherent_queue_batch_model.pt");

  std::ostringstream yaml;
  yaml << "name: incoherent_queue_batch\n";
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
  yaml << "max_batch_size: 8\n";
  yaml << "pool_size: 2\n";
  yaml << "max_queue_size: 4\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_incoherent_queue_batch.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: Incoherent batching config: max_queue_size (4) "
      "must be >= max_batch_size (8)";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, BatchingCoherenceRejectsInflightLowerThanPoolSize)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_incoherent_inflight_pool_model.pt");

  std::ostringstream yaml;
  yaml << "name: incoherent_inflight_pool\n";
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
  yaml << "max_batch_size: 4\n";
  yaml << "pool_size: 8\n";
  yaml << "max_queue_size: 32\n";
  yaml << "max_inflight_tasks: 3\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_incoherent_inflight_pool.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: Incoherent batching config: max_inflight_tasks "
      "(3) must be 0 (unbounded) or >= pool_size (8)";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, BatchingCoherenceAcceptsBoundaryValues)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_coherent_batching_model.pt");

  std::ostringstream yaml;
  yaml << "name: coherent_batching\n";
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
  yaml << "max_batch_size: 8\n";
  yaml << "pool_size: 4\n";
  yaml << "max_queue_size: 8\n";
  yaml << "max_inflight_tasks: 4\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_coherent_batching.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.batching.max_batch_size, 8);
  EXPECT_EQ(cfg.batching.pool_size, 4);
  EXPECT_EQ(cfg.batching.max_queue_size, 8U);
  EXPECT_EQ(cfg.batching.max_inflight_tasks, 4U);
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
max_queue_size: 0
max_batch_size: 1
batch_coalesce_timeout_ms: 1
pool_size: 1
)";
  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_no_input_skip.yaml";
  std::ofstream(tmp) << yaml;

  const RuntimeConfig cfg = load_config(tmp.string());
  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(cfg.batching.max_queue_size, kDefaultMaxQueueSize);
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
  const auto model_path =
      WriteEmptyModelFile("config_loader_overflow_dims_model.pt");

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
  ASSERT_TRUE(cfg.model.has_value());
  ASSERT_EQ(cfg.model->inputs.size(), 1U);
  EXPECT_EQ(
      cfg.model->inputs[0].dims,
      (std::vector<int64_t>{2147483647, 2147483647, 2147483647}));
  ASSERT_EQ(cfg.model->outputs.size(), 1U);

  // InvalidDimensionException cannot be triggered via YAML alone because
  // parse_tensor_nodes already rejects non-positive dimensions when reading the
  // config. A dedicated test uses the config loader post-parse hook to mutate
  // the parsed model and cover that catch.
}

TEST(ConfigLoader, UnsupportedDtypeDuringMaxMessageComputationMarksInvalid)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_complex_dtype_model.pt");

  std::ostringstream yaml;
  yaml << "name: complex_dtype\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: complex_input\n";
  yaml << "    dims: [1, 1]\n";
  yaml << "    data_type: float32\n";
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

  ConfigLoaderHookGuard hook_guard([](RuntimeConfig& cfg) {
    if (cfg.model.has_value() && !cfg.model->inputs.empty()) {
      cfg.model->inputs[0].type = at::kComplexFloat;
    }
  });

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: Unsupported at::ScalarType";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
  ASSERT_TRUE(cfg.model.has_value());
  ASSERT_EQ(cfg.model->inputs.size(), 1U);
  EXPECT_EQ(cfg.model->inputs[0].type, at::kComplexFloat);
  ASSERT_EQ(cfg.model->outputs.size(), 1U);
  EXPECT_FALSE(cfg.model->outputs.empty());
}

using VerbosityCase =
    ::testing::TestWithParam<std::pair<const char*, VerbosityLevel>>;

TEST_P(VerbosityCase, ParsesVerbosityStrings)
{
  const auto& [value, expected] = GetParam();
  const auto model_path =
      WriteEmptyModelFile(std::format("config_loader_verbosity_{}.pt", value));
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
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
