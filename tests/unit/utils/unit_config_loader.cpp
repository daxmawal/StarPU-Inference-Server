#include <gtest/gtest.h>

#include <array>
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

#include "support/utils/config_loader_test_api.hpp"
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

class ScopedCurrentPath {
 public:
  explicit ScopedCurrentPath(const std::filesystem::path& path)
      : previous_path_(std::filesystem::current_path())
  {
    std::filesystem::current_path(path);
  }

  ~ScopedCurrentPath()
  {
    std::error_code ec;
    std::filesystem::current_path(previous_path_, ec);
  }

  ScopedCurrentPath(const ScopedCurrentPath&) = delete;
  auto operator=(const ScopedCurrentPath&) -> ScopedCurrentPath& = delete;

 private:
  std::filesystem::path previous_path_;
};

auto
repository_root_path() -> std::filesystem::path
{
  return std::filesystem::weakly_canonical(
      std::filesystem::path(__FILE__).parent_path() / ".." / ".." / "..");
}

auto
adaptive_batching_yaml(
    int max_batch_size = 1,
    std::optional<int> min_batch_size = std::nullopt) -> std::string
{
  std::ostringstream yaml;
  yaml << "batching_strategy: adaptive\n";
  yaml << "adaptive_batching:\n";
  if (min_batch_size.has_value()) {
    yaml << "  min_batch_size: " << *min_batch_size << "\n";
  }
  yaml << "  max_batch_size: " << max_batch_size << "\n";
  return yaml.str();
}

auto
fixed_batching_yaml(int batch_size = 1) -> std::string
{
  std::ostringstream yaml;
  yaml << "batching_strategy: fixed\n";
  yaml << "fixed_batching:\n";
  yaml << "  batch_size: " << batch_size << "\n";
  return yaml.str();
}

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
             "batch_coalesce_timeout_ms: 1\n"} +
         adaptive_batching_yaml(1) + "pool_size: 1\n";
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
          yaml += "batch_coalesce_timeout_ms: 1\n";
          yaml += "batching_strategy: adaptive\n";
          yaml += "adaptive_batching:\n";
          yaml += "  max_batch_size: 0\n";
          yaml += "pool_size: 1\n";
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
        "LibtorchNotMapInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "libtorch: 1\n";
          return yaml;
        }(),
        "Failed to load config: libtorch must be a mapping"},
    InvalidConfigCase{
        "LibtorchIntraopThreadsNonPositiveInvalid",
        [] {
          auto yaml = base_model_yaml();
          yaml += "libtorch:\n";
          yaml += "  intraop_threads: 0\n";
          return yaml;
        }(),
        "Failed to load config: libtorch.intraop_threads must be > 0 and fit "
        "in int"},
    InvalidConfigCase{
        "LibtorchInteropThreadsTooLargeInvalid",
        [] {
          std::ostringstream yaml;
          yaml << base_model_yaml();
          yaml << "libtorch:\n";
          yaml << "  interop_threads: "
               << static_cast<long long>(std::numeric_limits<int>::max()) + 1
               << "\n";
          return yaml.str();
        }(),
        "Failed to load config: libtorch.interop_threads must be > 0 and fit "
        "in int"},
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
          yaml += adaptive_batching_yaml(1);
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
          yaml += adaptive_batching_yaml(1);
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
          yaml += adaptive_batching_yaml(1);
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
          yaml << adaptive_batching_yaml(1);
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
          yaml += adaptive_batching_yaml(1);
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
          yaml += adaptive_batching_yaml(1);
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
          yaml += adaptive_batching_yaml(1);
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
          yaml += adaptive_batching_yaml(1);
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
          yaml += adaptive_batching_yaml(1);
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
          yaml += adaptive_batching_yaml(1);
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
        "pool_size, batch_coalesce_timeout_ms, batching_strategy",
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
          yaml += adaptive_batching_yaml(1);
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
          yaml += adaptive_batching_yaml(1);
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
          yaml += adaptive_batching_yaml(1);
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
          yaml += adaptive_batching_yaml(1);
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
          yaml += adaptive_batching_yaml(1);
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
          yaml += adaptive_batching_yaml(1);
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
          yaml += adaptive_batching_yaml(1);
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
          yaml << adaptive_batching_yaml(1);
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
          yaml << adaptive_batching_yaml(1);
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

  starpu_server::CaptureStream capture{std::cerr};

  const RuntimeConfig cfg = load_config(config_path.string());

  if (test_case.expected_error.has_value()) {
    const std::string expected =
        expected_log_line(ErrorLevel, *test_case.expected_error);
    EXPECT_EQ(capture.str(), expected);
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

TEST(ConfigLoader, IOBlockParsesTensorNodesForTest)
{
  const YAML::Node nodes = YAML::Load(R"(
- name: first
  dims: [1, 4]
  data_type: float32
- name: second
  dims: [2]
  data_type: int64
)");

  const auto tensors = parse_tensor_nodes_for_test(nodes, 4U, "inputs", 4U);

  ASSERT_EQ(tensors.size(), 2U);
  EXPECT_EQ(tensors[0].name, "first");
  EXPECT_EQ(tensors[0].dims, (std::vector<int64_t>{1, 4}));
  EXPECT_EQ(tensors[0].type, at::kFloat);
  EXPECT_EQ(tensors[1].name, "second");
  EXPECT_EQ(tensors[1].dims, (std::vector<int64_t>{2}));
  EXPECT_EQ(tensors[1].type, at::kLong);
}

TEST(ConfigLoader, IOBlockRejectsTooManyTensorEntriesForTest)
{
  const YAML::Node nodes = YAML::Load(R"(
- { name: a, dims: [1], data_type: float32 }
- { name: b, dims: [1], data_type: float32 }
)");

  try {
    static_cast<void>(parse_tensor_nodes_for_test(nodes, 1U, "inputs", 4U));
    FAIL() << "Expected parse_tensor_nodes_for_test to reject too many entries";
  }
  catch (const std::invalid_argument& error) {
    EXPECT_EQ(error.what(), std::string("inputs must have at most 1 entries"));
  }
}

TEST(ConfigLoader, DeviceBlockParsesCpuCudaAndNumaSettings)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_device_block_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "use_cpu: false\n";
  yaml += "group_cpu_by_numa: true\n";
  yaml += "use_cuda:\n";
  yaml += "  - { device_ids: [2, 5] }\n";

  const auto config_path =
      WriteTempFile("config_loader_device_block.yaml", yaml);
  const RuntimeConfig cfg = load_config(config_path.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_FALSE(cfg.devices.use_cpu);
  EXPECT_TRUE(cfg.devices.group_cpu_by_numa);
  EXPECT_TRUE(cfg.devices.use_cuda);
  EXPECT_EQ(cfg.devices.ids, (std::vector<int>{2, 5}));
}

TEST(ConfigLoader, BatchingBlockParsesNetworkQueueAndTraceSettings)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_batching_block_model.pt");
  std::string yaml = ReplaceModelPath(
      std::string{"name: config_loader_test\n"
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
                  "batching_strategy: disabled\n"
                  "pool_size: 1\n"},
      model_path);
  const auto trace_dir = MakeUniqueTempDir("config_loader_batching_trace_dir");

  yaml += "address: 0.0.0.0:60061\n";
  yaml += "metrics_port: 9191\n";
  yaml += "max_message_bytes: 65536\n";
  yaml += "max_queue_size: 64\n";
  yaml += "max_inflight_tasks: 8\n";
  yaml += "trace_enabled: true\n";
  yaml += "trace_output: " + trace_dir.string() + "\n";

  const auto config_path =
      WriteTempFile("config_loader_batching_block.yaml", yaml);
  const RuntimeConfig cfg = load_config(config_path.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.server_address, "0.0.0.0:60061");
  EXPECT_EQ(cfg.metrics_port, 9191);
  EXPECT_EQ(cfg.batching.max_message_bytes, 65536U);
  EXPECT_EQ(cfg.batching.max_queue_size, 64U);
  EXPECT_EQ(cfg.batching.max_inflight_tasks, 8U);
  EXPECT_EQ(
      cfg.batching.strategy, starpu_server::BatchingStrategyKind::Disabled);
  EXPECT_TRUE(cfg.batching.trace_enabled);
  EXPECT_EQ(
      cfg.batching.trace_output_path,
      (trace_dir / std::string(kDefaultTraceFileName)).string());
}

TEST(ConfigLoader, CongestionBlockPartialConfigPreservesUnsetDefaults)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_congestion_partial_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += "  enabled: false\n";
  yaml += "  tick_interval_ms: 250\n";

  const auto config_path =
      WriteTempFile("config_loader_congestion_partial.yaml", yaml);
  const RuntimeConfig cfg = load_config(config_path.string());

  ASSERT_TRUE(cfg.valid);
  EXPECT_FALSE(cfg.congestion.enabled);
  EXPECT_EQ(cfg.congestion.tick_interval_ms, 250);
  EXPECT_DOUBLE_EQ(cfg.congestion.fill_high, kDefaultCongestionFillHigh);
  EXPECT_DOUBLE_EQ(cfg.congestion.fill_low, kDefaultCongestionFillLow);
  EXPECT_DOUBLE_EQ(cfg.congestion.alpha, kDefaultCongestionEwmaAlpha);
}

TEST(ConfigLoader, CongestionBlockParseHorizonsForTestAcceptsPositiveValues)
{
  YAML::Node congestion_node = YAML::Load(R"(
entry_horizon_ms: 250
exit_horizon_ms: 500
)");

  RuntimeConfig cfg;
  parse_congestion_horizons_for_test(congestion_node, cfg);

  EXPECT_EQ(cfg.congestion.entry_horizon_ms, 250);
  EXPECT_EQ(cfg.congestion.exit_horizon_ms, 500);
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

TEST(ConfigLoader, RejectsUseCudaEnabledWithoutDeviceIdsAfterParsing)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_cross_field_use_cuda_no_ids_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  const auto config_path =
      WriteTempFile("config_loader_cross_field_use_cuda_no_ids.yaml", yaml);

  ConfigLoaderHookGuard hook_guard([](RuntimeConfig& cfg) {
    cfg.devices.use_cuda = true;
    cfg.devices.ids.clear();
  });

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(config_path.string());

  const std::string expected_error =
      "Failed to load config: use_cuda is enabled but no CUDA device_ids are "
      "configured";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));
  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, RejectsDeviceIdsWhenUseCudaDisabledAfterParsing)
{
  const auto model_path = WriteEmptyModelFile(
      "config_loader_cross_field_ids_without_cuda_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  const auto config_path =
      WriteTempFile("config_loader_cross_field_ids_without_cuda.yaml", yaml);

  ConfigLoaderHookGuard hook_guard([](RuntimeConfig& cfg) {
    cfg.devices.use_cuda = false;
    cfg.devices.ids = {0};
  });

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(config_path.string());

  const std::string expected_error =
      "Failed to load config: device_ids are configured but use_cuda is "
      "disabled";
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
  yaml << "batching_strategy: adaptive\n";
  yaml << "adaptive_batching:\n";
  yaml << "  min_batch_size: 2\n";
  yaml << "  max_batch_size: 4\n";
  yaml << "pool_size: 2\n";
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
  EXPECT_EQ(cfg.batching.resolved_max_batch_size, 4);
  EXPECT_EQ(
      cfg.batching.strategy, starpu_server::BatchingStrategyKind::Adaptive);
  EXPECT_EQ(cfg.batching.adaptive.min_batch_size, 2);
  EXPECT_EQ(cfg.batching.adaptive.max_batch_size, 4);
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
  EXPECT_EQ(
      cfg.devices.gpu_model_replication,
      starpu_server::GpuModelReplicationPolicy::PerDevice);
}

TEST(ConfigLoader, ParsesGpuModelReplicationPolicy)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_gpu_model_replication_model.pt");

  std::ostringstream yaml;
  yaml << "name: gpu_model_replication\n";
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
  yaml << adaptive_batching_yaml(1);
  yaml << "pool_size: 1\n";
  yaml << "use_cuda:\n";
  yaml << "  - { device_ids: [0] }\n";
  yaml << "gpu_model_replication: per_worker\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_gpu_model_replication.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(
      cfg.devices.gpu_model_replication,
      starpu_server::GpuModelReplicationPolicy::PerWorker);
}

TEST(ConfigLoader, ParsesBatchingStrategy)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_batching_strategy_model.pt");

  std::ostringstream yaml;
  yaml << "name: batching_strategy\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "pool_size: 1\n";
  yaml << "batch_coalesce_timeout_ms: 5\n";
  yaml << "batching_strategy: fixed\n";
  yaml << "fixed_batching:\n";
  yaml << "  batch_size: 4\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_batching_strategy.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.batching.strategy, starpu_server::BatchingStrategyKind::Fixed);
  EXPECT_EQ(cfg.batching.fixed.batch_size, 4);
  EXPECT_EQ(cfg.batching.resolved_max_batch_size, 4);
}

TEST(ConfigLoader, RejectsInvalidBatchingStrategy)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_invalid_batching_strategy_model.pt");

  std::ostringstream yaml;
  yaml << "name: invalid_batching_strategy\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "pool_size: 1\n";
  yaml << "batch_coalesce_timeout_ms: 5\n";
  yaml << "batching_strategy: invalid_strategy\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_invalid_batching_strategy.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(
      capture.str(),
      expected_log_line(
          ErrorLevel,
          "Failed to load config: batching_strategy must be 'disabled', "
          "'adaptive' or 'fixed' (got 'invalid_strategy')"));
}

TEST(ConfigLoader, RejectsMissingBatchingStrategy)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_missing_batching_strategy_model.pt");

  std::ostringstream yaml;
  yaml << "name: missing_batching_strategy\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "pool_size: 1\n";
  yaml << "batch_coalesce_timeout_ms: 5\n";
  yaml << "adaptive_batching:\n";
  yaml << "  max_batch_size: 4\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_missing_batching_strategy.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(
      capture.str(),
      expected_log_line(
          ErrorLevel,
          "Failed to load config: Missing required key: batching_strategy"));
}

TEST(ConfigLoader, RejectsRemovedDynamicBatchingOption)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_removed_dynamic_batching_model.pt");

  std::ostringstream yaml;
  yaml << "name: removed_dynamic_batching\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "pool_size: 1\n";
  yaml << "batch_coalesce_timeout_ms: 5\n";
  yaml << "dynamic_batching: false\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_removed_dynamic_batching.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(
      capture.str(), expected_log_line(
                         ErrorLevel,
                         "Failed to load config: Unknown configuration option: "
                         "dynamic_batching"));
}

TEST(ConfigLoader, RejectsRemovedLegacyMaxBatchSizeOption)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_removed_legacy_batch_model.pt");

  std::ostringstream yaml;
  yaml << "name: removed_legacy_batch\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "max_batch_size: 4\n";
  yaml << "pool_size: 1\n";
  yaml << "batch_coalesce_timeout_ms: 5\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_removed_legacy_batch.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(
      capture.str(), expected_log_line(
                         ErrorLevel,
                         "Failed to load config: Unknown configuration option: "
                         "max_batch_size"));
}

TEST(ConfigLoader, RejectsAdaptiveBatchingBlockWhenStrategyIsFixed)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_fixed_with_adaptive_block_model.pt");

  std::ostringstream yaml;
  yaml << "name: fixed_with_adaptive_block\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "pool_size: 1\n";
  yaml << "batch_coalesce_timeout_ms: 5\n";
  yaml << "batching_strategy: fixed\n";
  yaml << "adaptive_batching:\n";
  yaml << "  max_batch_size: 4\n";
  yaml << "fixed_batching:\n";
  yaml << "  batch_size: 4\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_fixed_with_adaptive_block.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(
      capture.str(),
      expected_log_line(
          ErrorLevel,
          "Failed to load config: adaptive_batching cannot be used with "
          "batching_strategy 'fixed'"));
}

TEST(ConfigLoader, RejectsAdaptiveBatchingMinBatchAboveMaxBatch)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_invalid_adaptive_batching_model.pt");

  std::ostringstream yaml;
  yaml << "name: invalid_adaptive_batching\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "pool_size: 1\n";
  yaml << "batch_coalesce_timeout_ms: 5\n";
  yaml << "batching_strategy: adaptive\n";
  yaml << "adaptive_batching:\n";
  yaml << "  min_batch_size: 8\n";
  yaml << "  max_batch_size: 4\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_invalid_adaptive_batching.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(
      capture.str(),
      expected_log_line(
          ErrorLevel,
          "Failed to load config: adaptive_batching.min_batch_size must be <= "
          "adaptive_batching.max_batch_size"));
}

TEST(ConfigLoader, RejectsUnknownAdaptiveBatchingOption)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_unknown_adaptive_option_model.pt");

  std::ostringstream yaml;
  yaml << "name: unknown_adaptive_option\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "pool_size: 1\n";
  yaml << "batch_coalesce_timeout_ms: 5\n";
  yaml << "batching_strategy: adaptive\n";
  yaml << "adaptive_batching:\n";
  yaml << "  max_batch_size: 4\n";
  yaml << "  unexpected: 1\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_unknown_adaptive_option.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(
      capture.str(), expected_log_line(
                         ErrorLevel,
                         "Failed to load config: Unknown configuration option: "
                         "adaptive_batching.unexpected"));
}

TEST(ConfigLoader, RejectsAdaptiveStrategyWithoutAdaptiveBatchingBlock)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_missing_adaptive_block_model.pt");

  std::ostringstream yaml;
  yaml << "name: missing_adaptive_block\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "pool_size: 1\n";
  yaml << "batch_coalesce_timeout_ms: 5\n";
  yaml << "batching_strategy: adaptive\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_missing_adaptive_block.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(
      capture.str(),
      expected_log_line(
          ErrorLevel,
          "Failed to load config: batching_strategy 'adaptive' requires "
          "adaptive_batching.max_batch_size"));
}

TEST(ConfigLoader, RejectsFixedStrategyWithoutFixedBatchingBlock)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_missing_fixed_block_model.pt");

  std::ostringstream yaml;
  yaml << "name: missing_fixed_block\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "pool_size: 1\n";
  yaml << "batch_coalesce_timeout_ms: 5\n";
  yaml << "batching_strategy: fixed\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_missing_fixed_block.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(
      capture.str(),
      expected_log_line(
          ErrorLevel,
          "Failed to load config: batching_strategy 'fixed' requires "
          "fixed_batching.batch_size"));
}

TEST(ConfigLoader, RejectsUnknownFixedBatchingOption)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_unknown_fixed_option_model.pt");

  std::ostringstream yaml;
  yaml << "name: unknown_fixed_option\n";
  yaml << "model: " << model_path.string() << "\n";
  yaml << "inputs:\n";
  yaml << "  - name: in\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "outputs:\n";
  yaml << "  - name: out\n";
  yaml << "    dims: [1]\n";
  yaml << "    data_type: float32\n";
  yaml << "pool_size: 1\n";
  yaml << "batch_coalesce_timeout_ms: 5\n";
  yaml << "batching_strategy: fixed\n";
  yaml << "fixed_batching:\n";
  yaml << "  batch_size: 4\n";
  yaml << "  unexpected: 1\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_unknown_fixed_option.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(
      capture.str(), expected_log_line(
                         ErrorLevel,
                         "Failed to load config: Unknown configuration option: "
                         "fixed_batching.unexpected"));
}

TEST(ConfigLoader, RejectsInvalidGpuModelReplicationPolicy)
{
  const auto model_path = WriteEmptyModelFile(
      "config_loader_invalid_gpu_model_replication_model.pt");

  std::ostringstream yaml;
  yaml << "name: invalid_gpu_model_replication\n";
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
  yaml << adaptive_batching_yaml(1);
  yaml << "pool_size: 1\n";
  yaml << "gpu_model_replication: invalid_policy\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_invalid_gpu_model_replication.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(
      capture.str(),
      expected_log_line(
          ErrorLevel,
          "Failed to load config: gpu_model_replication must be 'per_device' "
          "or 'per_worker' (got 'invalid_policy')"));
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
  yaml << adaptive_batching_yaml(1);
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
  yaml << adaptive_batching_yaml(1);
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
  yaml << adaptive_batching_yaml(1);
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
  yaml << adaptive_batching_yaml(1);
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
  yaml << adaptive_batching_yaml(1);
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
  yaml << adaptive_batching_yaml(1);
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
  yaml << "batching_strategy: adaptive\n";
  yaml << "adaptive_batching:\n";
  yaml << "  max_batch_size: 0\n";
  yaml << "batch_coalesce_timeout_ms: 1\n";
  yaml << "pool_size: 1\n";

  const auto tmp =
      std::filesystem::temp_directory_path() / "config_loader_zero_batch.yaml";
  std::ofstream(tmp) << yaml.str();

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: adaptive_batching.max_batch_size must be > 0 and "
      "fit in int";
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
  yaml << adaptive_batching_yaml(1);
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
  yaml << adaptive_batching_yaml(1);
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
  yaml << adaptive_batching_yaml(2);
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
  yaml << adaptive_batching_yaml(1);
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
  yaml << adaptive_batching_yaml(1);
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
  yaml << adaptive_batching_yaml(1);
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
  yaml << adaptive_batching_yaml(1);
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

TEST(ConfigLoader, CongestionLatencySloRejectsNegative)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_negative_latency_slo_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += "  latency_slo_ms: -1\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_negative_latency_slo.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: latency_slo_ms must be >= 0";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, CongestionQueueLatencyBudgetMsRejectsNegative)
{
  const auto model_path = WriteEmptyModelFile(
      "config_loader_negative_queue_latency_budget_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += "  queue_latency_budget_ms: -1\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_negative_queue_latency_budget.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: queue_latency_budget_ms must be >= 0";
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

TEST(ConfigLoader, CongestionEntryHorizonRejectsNonPositive)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_entry_horizon_nonpositive_model.pt");

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += "  entry_horizon_ms: 0\n";
  yaml += "  exit_horizon_ms: 10\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_entry_horizon_nonpositive.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: entry_horizon_ms must be > 0 and fit in int";
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, expected_error));

  EXPECT_FALSE(cfg.valid);
}

TEST(ConfigLoader, CongestionEntryHorizonRejectsValueAboveIntMax)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_entry_horizon_too_large_model.pt");
  const auto too_large =
      static_cast<long long>(std::numeric_limits<int>::max()) + 1LL;

  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "congestion:\n";
  yaml += std::format("  entry_horizon_ms: {}\n", too_large);
  yaml += "  exit_horizon_ms: 10\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_entry_horizon_too_large.yaml";
  std::ofstream(tmp) << yaml;

  starpu_server::CaptureStream capture{std::cerr};
  const RuntimeConfig cfg = load_config(tmp.string());

  const std::string expected_error =
      "Failed to load config: entry_horizon_ms must be > 0 and fit in int";
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
  yaml << adaptive_batching_yaml(1);
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
  yaml << adaptive_batching_yaml(8);
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
  yaml << adaptive_batching_yaml(4);
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
  yaml << adaptive_batching_yaml(8);
  yaml << "pool_size: 4\n";
  yaml << "max_queue_size: 8\n";
  yaml << "max_inflight_tasks: 4\n";

  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_coherent_batching.yaml";
  std::ofstream(tmp) << yaml.str();

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.batching.resolved_max_batch_size, 8);
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
batch_coalesce_timeout_ms: 1
batching_strategy: disabled
pool_size: 1
)";
  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_no_model_skip.yaml";
  std::ofstream(tmp) << yaml;

  RuntimeConfig cfg;
  std::string log;
  {
    starpu_server::CaptureStream capture{std::cerr};
    cfg = load_config(tmp.string());
    log = capture.str();
  }
  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(cfg.batching.resolved_max_batch_size, 1);
  EXPECT_NE(log.find("Missing required key: model"), std::string::npos);
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
batch_coalesce_timeout_ms: 1
batching_strategy: disabled
pool_size: 1
)";
  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_no_input_skip.yaml";
  std::ofstream(tmp) << yaml;

  RuntimeConfig cfg;
  std::string log;
  {
    starpu_server::CaptureStream capture{std::cerr};
    cfg = load_config(tmp.string());
    log = capture.str();
  }
  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(cfg.batching.max_queue_size, kDefaultMaxQueueSize);
  EXPECT_NE(log.find("Missing required key: inputs"), std::string::npos);
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
batch_coalesce_timeout_ms: 1
batching_strategy: disabled
pool_size: 1
)";
  const auto tmp = std::filesystem::temp_directory_path() /
                   "config_loader_no_output_skip.yaml";
  std::ofstream(tmp) << yaml;

  RuntimeConfig cfg;
  std::string log;
  {
    starpu_server::CaptureStream capture{std::cerr};
    cfg = load_config(tmp.string());
    log = capture.str();
  }
  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(cfg.batching.resolved_max_batch_size, 1);
  EXPECT_NE(log.find("Missing required key: outputs"), std::string::npos);
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
  yaml << adaptive_batching_yaml(1);
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
  yaml << adaptive_batching_yaml(1);
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

TEST(ConfigLoader, LoadsVersionedExampleYamlFiles)
{
  const auto repo_root = repository_root_path();
  const auto working_dir =
      repo_root /
      std::format(".config_loader_example_cwd_{}", NextTempArtifactSequence());
  std::filesystem::create_directories(working_dir);
  ScopedPermissionRestorer working_dir_cleanup(working_dir);
  ScopedCurrentPath scoped_current_path(working_dir);

  const std::array example_paths{
      repo_root / "models/bert.yml",
      repo_root / "models/bert_docker.yml",
      repo_root / "models/resnet18.yml",
      repo_root / "models/resnet18_fixed.yml",
      repo_root / "models/resnet152.yml",
      repo_root / "models/vit_l_16.yml",
      repo_root / "ci/perf/resnet152_ci_perf.yml",
      repo_root / "ci/perf/resnet152_ci_perf_gpu_only.yml",
      repo_root / "ci/perf/resnet152_ci_perf_fixed_gpu_only.yml",
  };

  for (const auto& config_path : example_paths) {
    SCOPED_TRACE(config_path.string());
    ASSERT_TRUE(std::filesystem::exists(config_path));

    auto effective_config_path = config_path;
    if (config_path.filename() == "bert_docker.yml") {
      std::ifstream input(config_path);
      ASSERT_TRUE(input.is_open());

      std::ostringstream normalized_yaml;
      normalized_yaml << input.rdbuf();

      const std::string docker_model_path =
          "/workspace/models/bert_libtorch.pt";
      const std::string test_model_path =
          (repo_root / "models/bert_libtorch.pt").string();
      const auto path_pos = normalized_yaml.str().find(docker_model_path);
      ASSERT_NE(path_pos, std::string::npos);

      std::string yaml_text = normalized_yaml.str();
      yaml_text.replace(path_pos, docker_model_path.size(), test_model_path);
      effective_config_path =
          WriteTempFile("config_loader_bert_docker_example.yaml", yaml_text);
    }

    const RuntimeConfig cfg = load_config(effective_config_path.string());

    EXPECT_TRUE(cfg.valid);
    EXPECT_FALSE(cfg.name.empty());
    ASSERT_TRUE(cfg.model.has_value());
    EXPECT_FALSE(cfg.model->path.empty());
  }
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

TEST(ConfigLoader, ParsesLibtorchThreadSettings)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_libtorch_thread_settings_model.pt");
  std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);
  yaml += "libtorch:\n";
  yaml += "  intraop_threads: 2\n";
  yaml += "  interop_threads: 3\n";

  const auto tmp =
      WriteTempFile("config_loader_libtorch_thread_settings.yaml", yaml);

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  ASSERT_TRUE(cfg.libtorch.intraop_threads.has_value());
  ASSERT_TRUE(cfg.libtorch.interop_threads.has_value());
  EXPECT_EQ(*cfg.libtorch.intraop_threads, 2);
  EXPECT_EQ(*cfg.libtorch.interop_threads, 3);
}

TEST(ConfigLoader, LeavesLibtorchThreadSettingsUnsetByDefault)
{
  const auto model_path =
      WriteEmptyModelFile("config_loader_default_libtorch_threads_model.pt");
  const std::string yaml = ReplaceModelPath(base_model_yaml(), model_path);

  const auto tmp =
      WriteTempFile("config_loader_default_libtorch_threads.yaml", yaml);

  const RuntimeConfig cfg = load_config(tmp.string());

  EXPECT_TRUE(cfg.valid);
  EXPECT_FALSE(cfg.libtorch.intraop_threads.has_value());
  EXPECT_FALSE(cfg.libtorch.interop_threads.has_value());
}
