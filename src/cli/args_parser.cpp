#include "args_parser.hpp"

#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "datatype_utils.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"

namespace starpu_server {
constexpr int kPortMin = 1;
constexpr int kPortMax = 65535;

// =============================================================================
// Shape and Type Parsing: Handle --shape, --shapes, and --types arguments
// =============================================================================

static auto
parse_shape_string(const std::string& shape_str) -> std::vector<int64_t>
{
  std::vector<int64_t> shape;
  std::stringstream shape_stream(shape_str);
  std::string item;

  while (std::getline(shape_stream, item, 'x')) {
    int64_t dim = 0;
    try {
      dim = std::stoll(item);
    }
    catch (const std::invalid_argument& e) {
      throw std::invalid_argument(
          "Shape contains non-integer: " + std::string(e.what()));
    }
    catch (const std::out_of_range& e) {
      throw std::out_of_range(
          "Shape value out of range: " + std::string(e.what()));
    }

    if (dim <= 0) {
      throw std::invalid_argument("Shape dimension must be positive.");
    }

    shape.push_back(dim);
  }

  if (shape.empty()) {
    throw std::invalid_argument("Shape string is empty or invalid.");
  }

  return shape;
}

static auto
parse_shapes_string(const std::string& shapes_str)
    -> std::vector<std::vector<int64_t>>
{
  std::vector<std::vector<int64_t>> shapes;
  std::stringstream shape_stream(shapes_str);
  std::string shape_str;

  if (!shapes_str.empty() && shapes_str.back() == ',') {
    throw std::invalid_argument("Trailing comma in shapes string");
  }

  while (std::getline(shape_stream, shape_str, ',')) {
    if (shape_str.empty()) {
      throw std::invalid_argument("Empty shape in shapes string");
    }
    shapes.push_back(parse_shape_string(shape_str));
  }

  if (shapes.empty()) {
    throw std::invalid_argument("No valid shapes provided.");
  }

  return shapes;
}

static auto
parse_types_string(const std::string& types_str) -> std::vector<at::ScalarType>
{
  if (types_str.empty()) {
    throw std::invalid_argument("No types provided.");
  }

  if (types_str.find_first_not_of(',') == std::string::npos) {
    throw std::invalid_argument("No types provided.");
  }

  std::vector<at::ScalarType> types;
  std::stringstream type_stream(types_str);
  std::string type_str;

  if (types_str.back() == ',') {
    throw std::invalid_argument("Trailing comma in types string");
  }

  while (std::getline(type_stream, type_str, ',')) {
    if (type_str.empty()) {
      throw std::invalid_argument("Empty type in types string");
    }
    try {
      types.push_back(string_to_scalar_type(type_str));
    }
    catch (const std::invalid_argument& e) {
      throw std::invalid_argument(std::string{"Unsupported type: "} + type_str);
    }
  }

  if (types.empty()) {
    throw std::invalid_argument("No types provided.");
  }

  return types;
}

// =============================================================================
// Combined --input parser: "name:D1xD2x...:dtype" or "D1xD2x...:dtype"
// =============================================================================

static auto missing_value_error(std::string_view option_name) -> bool;

static auto
parse_input_combined(RuntimeConfig& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  if (idx + 1 >= args.size()) {
    return missing_value_error("--input");
  }
  ++idx;

  std::string spec = args[idx];

  std::vector<std::string> parts;
  {
    std::stringstream spec_stream(spec);
    std::string part;
    while (std::getline(spec_stream, part, ':')) {
      parts.push_back(part);
    }
  }

  if (parts.size() != 2 && parts.size() != 3) {
    log_error(std::format(
        "Invalid --input format: {}. Expected name:DIMS:TYPE or DIMS:TYPE",
        spec));
    return false;
  }

  std::string name;
  std::string dims_str;
  std::string type_str;

  if (parts.size() == 3) {
    name = parts[0];
    dims_str = parts[1];
    type_str = parts[2];
  } else {
    dims_str = parts[0];
    type_str = parts[1];
  }

  std::vector<int64_t> dims;
  at::ScalarType dtype = at::ScalarType::Undefined;
  try {
    dims = parse_shape_string(dims_str);
    dtype = string_to_scalar_type(type_str);
  }
  catch (const std::exception& e) {
    log_error(std::format("Invalid --input '{}': {}", spec, e.what()));
    return false;
  }

  opts.models.resize(1);
  if (!opts.seen_combined_input) {
    opts.models[0].inputs.clear();
    opts.seen_combined_input = true;
  }

  TensorConfig tensor_config{};
  tensor_config.name = name.empty()
                           ? (std::string("input") +
                              std::to_string(opts.models[0].inputs.size()))
                           : name;
  tensor_config.dims = std::move(dims);
  tensor_config.type = dtype;
  opts.models[0].inputs.push_back(std::move(tensor_config));

  return true;
}

// =============================================================================
// Argument Parsing Utilities: Helpers for optional and positional argument
// parsing
// =============================================================================

static void
check_required(
    const bool condition, const std::string& option_name,
    std::vector<std::string>& missing)
{
  if (!condition) {
    missing.push_back(option_name);
  }
}

static auto
missing_value_error(std::string_view option_name) -> bool
{
  log_error(std::format("{} option requires a value.", option_name));
  return false;
}

template <typename Func>
auto
try_parse(const char* val, Func&& parser) -> bool
{
  try {
    std::forward<Func>(parser)(val);
    return true;
  }
  catch (const std::invalid_argument& e) {
    log_error(e.what());
  }
  catch (const std::out_of_range& e) {
    log_error(e.what());
  }
  return false;
}

template <typename Func>
auto
expect_and_parse(
    std::string_view option_name, size_t& idx, std::span<char*> args,
    Func&& parser) -> bool
{
  if (idx + 1 >= args.size()) {
    return missing_value_error(option_name);
  }
  ++idx;
  return try_parse(args[idx], std::forward<Func>(parser));
}

// =============================================================================
// Individual Argument Parsers for --model, --shape, etc.
// =============================================================================

static auto
parse_model(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  if (idx + 1 >= args.size()) {
    return missing_value_error("--model");
  }
  ++idx;
  opts.models.resize(1);
  opts.models[0].path = args[idx];
  if (!std::filesystem::exists(opts.models[0].path)) {
    log_error(std::format("Model file not found: {}", opts.models[0].path));
    return false;
  }
  return true;
}

static auto
parse_config(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  if (idx + 1 >= args.size()) {
    return missing_value_error("--config");
  }
  ++idx;
  opts.config_path = args[idx];
  if (!std::filesystem::exists(opts.config_path)) {
    log_error(std::format("Config file not found: {}", opts.config_path));
    return false;
  }
  return true;
}

static auto
parse_iterations(RuntimeConfig& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  auto& iterations = opts.iterations;
  return expect_and_parse(
      "--iterations", idx, args, [&iterations](const char* val) {
        const auto tmp = std::stoi(val);
        if (tmp <= 0) {
          throw std::invalid_argument("Must be > 0.");
        }
        iterations = tmp;
      });
}

static auto
parse_shape(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  opts.models.resize(1);
  auto& inputs = opts.models[0].inputs;
  return expect_and_parse("--shape", idx, args, [&inputs](const char* val) {
    auto dims = parse_shape_string(val);
    inputs.resize(1);
    inputs[0].name = inputs[0].name.empty() ? "input0" : inputs[0].name;
    inputs[0].dims = std::move(dims);
  });
}

static auto
parse_shapes(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  opts.models.resize(1);
  auto& inputs = opts.models[0].inputs;
  return expect_and_parse("--shapes", idx, args, [&inputs](const char* val) {
    auto dims_list = parse_shapes_string(val);
    inputs.resize(dims_list.size());
    for (size_t i = 0; i < dims_list.size(); ++i) {
      inputs[i].name = inputs[i].name.empty()
                           ? std::string("input") + std::to_string(i)
                           : inputs[i].name;
      inputs[i].dims = std::move(dims_list[i]);
    }
  });
}

static auto
parse_types(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  opts.models.resize(1);
  auto& inputs = opts.models[0].inputs;
  return expect_and_parse("--types", idx, args, [&inputs](const char* val) {
    auto types = parse_types_string(val);
    if (inputs.size() < types.size()) {
      inputs.resize(types.size());
    }
    for (size_t i = 0; i < types.size(); ++i) {
      inputs[i].name = inputs[i].name.empty()
                           ? std::string("input") + std::to_string(i)
                           : inputs[i].name;
      inputs[i].type = types[i];
    }
  });
}

static auto
parse_verbose(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  auto& verbosity = opts.verbosity;
  return expect_and_parse(
      "--verbose", idx, args, [&verbosity](const char* val) {
        verbosity = parse_verbosity_level(val);
      });
}

static auto
parse_delay(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  auto& delay_ms = opts.delay_ms;
  return expect_and_parse("--delay", idx, args, [&delay_ms](const char* val) {
    delay_ms = std::stoi(val);
    if (delay_ms < 0) {
      throw std::invalid_argument("Must be >= 0.");
    }
  });
}

static auto
parse_device_ids(RuntimeConfig& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  const bool parsed =
      expect_and_parse("--device-ids", idx, args, [&opts](const char* val) {
        opts.use_cuda = true;
        std::stringstream shape_stream(val);
        std::string id_str;
        while (std::getline(shape_stream, id_str, ',')) {
          const int device_id = std::stoi(id_str);
          if (device_id < 0) {
            throw std::invalid_argument("Must be >= 0.");
          }
          opts.device_ids.push_back(device_id);
        }
        if (opts.device_ids.empty()) {
          throw std::invalid_argument("No device IDs provided.");
        }
      });

  if (!parsed) {
    return false;
  }

  std::unordered_set<int> unique_ids(
      opts.device_ids.begin(), opts.device_ids.end());
  if (unique_ids.size() != opts.device_ids.size()) {
    log_error("Duplicate device IDs provided.");
    return false;
  }

  const int device_count =
      static_cast<int>(static_cast<unsigned char>(torch::cuda::device_count()));

  const auto invalid_it = std::ranges::find_if(
      opts.device_ids, [device_count](const int device_id) noexcept {
        return device_id >= device_count;
      });
  if (invalid_it != opts.device_ids.end()) {
    log_error(std::format(
        "GPU ID {} out of range. Only {} device(s) available.", *invalid_it,
        device_count));
    return false;
  }

  return true;
}

static auto
parse_address(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  if (idx + 1 >= args.size()) {
    return missing_value_error("--address");
  }
  ++idx;
  opts.server_address = args[idx];
  return true;
}

static auto
parse_max_batch_size(RuntimeConfig& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  auto& max_batch_size = opts.max_batch_size;
  return expect_and_parse(
      "--max-batch-size", idx, args, [&max_batch_size](const char* val) {
        const int tmp = std::stoi(val);
        if (tmp <= 0) {
          throw std::invalid_argument("Must be > 0.");
        }
        max_batch_size = tmp;
      });
}

static auto
parse_input_slots(RuntimeConfig& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  auto& input_slots = opts.input_slots;
  return expect_and_parse(
      "--input-slots", idx, args, [&input_slots](const char* val) {
        const int tmp = std::stoi(val);
        if (tmp <= 0) {
          throw std::invalid_argument("Must be > 0.");
        }
        input_slots = tmp;
      });
}

static auto
parse_slots_alias(RuntimeConfig& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  auto& input_slots = opts.input_slots;
  return expect_and_parse(
      "--slots", idx, args, [&input_slots](const char* val) {
        const int tmp = std::stoi(val);
        if (tmp <= 0) {
          throw std::invalid_argument("Must be > 0.");
        }
        input_slots = tmp;
      });
}

static auto
parse_metrics_port(RuntimeConfig& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  auto& metrics_port = opts.metrics_port;
  return expect_and_parse(
      "--metrics-port", idx, args, [&metrics_port](const char* val) {
        metrics_port = std::stoi(val);
        if (metrics_port < kPortMin || metrics_port > kPortMax) {
          throw std::out_of_range("Metrics port must be between 1 and 65535.");
        }
      });
}

static auto
parse_scheduler(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  if (idx + 1 >= args.size()) {
    return missing_value_error("--scheduler");
  }
  ++idx;
  const std::string scheduler = args[idx];
  if (!kAllowedSchedulers.contains(scheduler)) {
    std::ostringstream oss;
    for (auto it = kAllowedSchedulers.begin(); it != kAllowedSchedulers.end();
         ++it) {
      if (it != kAllowedSchedulers.begin()) {
        oss << ", ";
      }
      oss << *it;
    }
    log_error(std::format(
        "Unknown scheduler: '{}'. Allowed schedulers: {}", scheduler,
        oss.str()));
    return false;
  }
  opts.scheduler = scheduler;
  return true;
}

static auto
parse_pregen_inputs(RuntimeConfig& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  auto& pregen = opts.pregen_inputs;
  return expect_and_parse(
      "--pregen-inputs", idx, args, [&pregen](const char* val) {
        const auto tmp = std::stoi(val);
        if (tmp <= 0) {
          throw std::invalid_argument("Must be > 0.");
        }
        pregen = static_cast<size_t>(tmp);
      });
}

static auto
parse_warmup_pregen_inputs(
    RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  auto& pregen = opts.warmup_pregen_inputs;
  return expect_and_parse(
      "--warmup-pregen-inputs", idx, args, [&pregen](const char* val) {
        const auto tmp = std::stoi(val);
        if (tmp <= 0) {
          throw std::invalid_argument("Must be > 0.");
        }
        pregen = static_cast<size_t>(tmp);
      });
}

static auto
parse_warmup_iterations(RuntimeConfig& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  auto& warmup = opts.warmup_iterations;
  return expect_and_parse(
      "--warmup-iterations", idx, args, [&warmup](const char* val) {
        const auto tmp = std::stoi(val);
        if (tmp < 0) {
          throw std::invalid_argument("Must be >= 0.");
        }
        warmup = tmp;
      });
}

static auto
parse_seed(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  auto& seed = opts.seed;
  return expect_and_parse("--seed", idx, args, [&seed](const char* val) {
    const auto tmp = std::stoll(val);
    if (tmp < 0) {
      throw std::invalid_argument("Must be >= 0.");
    }
    seed = static_cast<uint64_t>(tmp);
  });
}

static auto
parse_rtol(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  auto& rtol = opts.rtol;
  return expect_and_parse("--rtol", idx, args, [&rtol](const char* val) {
    const auto tmp = std::stod(val);
    if (tmp < 0) {
      throw std::invalid_argument("Must be >= 0.");
    }
    rtol = tmp;
  });
}

static auto
parse_atol(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  auto& atol = opts.atol;
  return expect_and_parse("--atol", idx, args, [&atol](const char* val) {
    const auto tmp = std::stod(val);
    if (tmp < 0) {
      throw std::invalid_argument("Must be >= 0.");
    }
    atol = tmp;
  });
}

// =============================================================================
// Dispatch Argument Parser (Main parser loop)
// =============================================================================
struct TransparentEqual {
  using is_transparent = void;
  auto operator()(std::string_view lhs, std::string_view rhs) const noexcept
      -> bool
  {
    return lhs == rhs;
  }
};

static auto
parse_argument_values(std::span<char*> args_span, RuntimeConfig& opts) -> bool
{
  using Parser = bool (*)(RuntimeConfig&, size_t&, std::span<char*>);
  const static std::unordered_map<
      std::string_view, Parser, TransparentHash, TransparentEqual>
      dispatch = {
          {"--input", parse_input_combined},
          {"--config", parse_config},
          {"-c", parse_config},
          {"--model", parse_model},
          {"--iterations", parse_iterations},
          {"--shape", parse_shape},
          {"--shapes", parse_shapes},
          {"--types", parse_types},
          {"--verbose", parse_verbose},
          {"--delay", parse_delay},
          {"--device-ids", parse_device_ids},
          {"--scheduler", parse_scheduler},
          {"--address", parse_address},
          {"--metrics-port", parse_metrics_port},
          {"--max-batch-size", parse_max_batch_size},
          {"--input-slots", parse_input_slots},
          {"--slots", parse_slots_alias},
          {"--pregen-inputs", parse_pregen_inputs},
          {"--warmup-pregen-inputs", parse_warmup_pregen_inputs},
          {"--warmup-iterations", parse_warmup_iterations},
          {"--seed", parse_seed},
          {"--rtol", parse_rtol},
          {"--atol", parse_atol},
      };

  for (size_t idx = 1; idx < args_span.size(); ++idx) {
    const std::string_view arg = args_span[idx];

    if (arg == "--sync") {
      opts.synchronous = true;
    } else if (arg == "--no_cpu") {
      opts.use_cpu = false;
    } else if (arg == "--no-validate") {
      opts.validate_results = false;
    } else if (arg == "--help" || arg == "-h") {
      opts.show_help = true;
      return true;
    } else if (auto iter = dispatch.find(arg); iter != dispatch.end()) {
      if (!iter->second(opts, idx, args_span)) {
        return false;
      }
    } else {
      log_error(std::format(
          "Unknown argument: {}. Use --help to see valid options.", arg));
      return false;
    }
  }

  return true;
}

// =============================================================================
// Config Validation: Ensures all required fields are present and consistent
// =============================================================================

static auto
validate_config(RuntimeConfig& opts) -> void
{
  std::vector<std::string> missing;
  check_required(
      !opts.models.empty() && !opts.models[0].path.empty(), "--model", missing);
  const bool have_shapes =
      !opts.models.empty() &&
      std::any_of(
          opts.models[0].inputs.begin(), opts.models[0].inputs.end(),
          [](const auto& tensor) { return !tensor.dims.empty(); });
  const bool have_types =
      !opts.models.empty() &&
      std::all_of(
          opts.models[0].inputs.begin(), opts.models[0].inputs.end(),
          [](const auto& tensor) {
            return tensor.type != at::ScalarType::Undefined;
          });
  check_required(have_shapes, "--shape or --shapes", missing);
  check_required(have_types, "--types", missing);

  if (have_shapes && have_types) {
    for (const auto& input : opts.models[0].inputs) {
      if (input.dims.empty() || input.type == at::ScalarType::Undefined) {
        log_error("Number of --types must match number of input shapes.");
        opts.valid = false;
        break;
      }
    }
  }

  if (!missing.empty()) {
    for (const auto& opt : missing) {
      log_error(std::format("{} option is required.", opt));
    }
    opts.valid = false;
  }
}

// =============================================================================
// Top-Level Entry: Parses all arguments into a RuntimeConfig object
// =============================================================================

auto
parse_arguments(std::span<char*> args_span, RuntimeConfig opts) -> RuntimeConfig
{
  if (!parse_argument_values(args_span, opts)) {
    opts.valid = false;
    return opts;
  }

  if (!opts.show_help) {
    validate_config(opts);
    if (opts.valid) {
      try {
        opts.max_message_bytes = compute_max_message_bytes(
            opts.max_batch_size, opts.models, opts.max_message_bytes);
      }
      catch (const InvalidDimensionException& e) {
        log_error(e.what());
        opts.valid = false;
      }
      catch (const MessageSizeOverflowException& e) {
        log_error(e.what());
        opts.valid = false;
      }
      catch (const UnsupportedDtypeException& e) {
        log_error(e.what());
        opts.valid = false;
      }
    }
  }

  return opts;
}

}  // namespace starpu_server
