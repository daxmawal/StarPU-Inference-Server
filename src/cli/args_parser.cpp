#include "args_parser.hpp"

#include <c10/core/ScalarType.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
#include <functional>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "logger.hpp"
#include "runtime_config.hpp"
#include "transparent_hash.hpp"

namespace starpu_server {

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
parse_type_string(const std::string& type_str) -> at::ScalarType
{
  static const std::unordered_map<
      std::string, at::ScalarType, TransparentHash, std::equal_to<>>
      type_map = {
          {"float", at::kFloat},
          {"float32", at::kFloat},
          {"double", at::kDouble},
          {"float64", at::kDouble},
          {"half", at::kHalf},
          {"float16", at::kHalf},
          {"bfloat16", at::kBFloat16},
          {"int", at::kInt},
          {"int32", at::kInt},
          {"long", at::kLong},
          {"int64", at::kLong},
          {"short", at::kShort},
          {"int16", at::kShort},
          {"char", at::kChar},
          {"int8", at::kChar},
          {"byte", at::kByte},
          {"uint8", at::kByte},
          {"bool", at::kBool},
          {"complex64", at::kComplexFloat},
          {"complex128", at::kComplexDouble},
      };

  std::string type_lower = type_str;
  std::transform(
      type_lower.begin(), type_lower.end(), type_lower.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  auto iterator = type_map.find(type_lower);
  if (iterator == type_map.end()) {
    throw std::invalid_argument("Unsupported type: " + type_str);
  }
  return iterator->second;
}

static auto
parse_types_string(const std::string& types_str) -> std::vector<at::ScalarType>
{
  if (types_str.empty()) {
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
    types.push_back(parse_type_string(type_str));
  }

  if (types.empty()) {
    throw std::invalid_argument("No types provided.");
  }

  return types;
}

// =============================================================================
// Verbosity Parsing
// =============================================================================

static auto
parse_verbosity_level(const std::string& val) -> VerbosityLevel
{
  using enum VerbosityLevel;
  const int level = std::stoi(val);
  switch (level) {
    case 0:
      return Silent;
    case 1:
      return Info;
    case 2:
      return Stats;
    case 3:
      return Debug;
    case 4:
      return Trace;
    default:
      throw std::invalid_argument("Invalid verbosity level: " + val);
  }
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

template <typename Func>
auto
try_parse(const char* val, Func&& parser) -> bool
{
  try {
    std::forward<Func>(parser)(val);
    return true;
  }
  catch (const std::invalid_argument& e) {
    log_error(std::format("Invalid argument: {}", e.what()));
  }
  catch (const std::out_of_range& e) {
    log_error(std::format("Out of range: {}", e.what()));
  }
  return false;
}

template <typename Func>
auto
expect_and_parse(size_t& idx, std::span<char*> args, Func&& parser) -> bool
{
  if (idx + 1 >= args.size()) {
    return false;
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
    return false;
  }
  ++idx;
  opts.model_path = args[idx];
  return true;
}

static auto
parse_config(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  if (idx + 1 >= args.size()) {
    return false;
  }
  ++idx;
  opts.config_path = args[idx];
  return true;
}

static auto
parse_iterations(RuntimeConfig& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  auto& iterations = opts.iterations;
  return expect_and_parse(idx, args, [&iterations](const char* val) {
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
  auto& inputs = opts.inputs;
  return expect_and_parse(idx, args, [&inputs](const char* val) {
    auto dims = parse_shape_string(val);
    inputs.resize(1);
    inputs[0].name = inputs[0].name.empty() ? "input0" : inputs[0].name;
    inputs[0].dims = std::move(dims);
  });
}

static auto
parse_shapes(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  auto& inputs = opts.inputs;
  return expect_and_parse(idx, args, [&inputs](const char* val) {
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
  auto& inputs = opts.inputs;
  return expect_and_parse(idx, args, [&inputs](const char* val) {
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
  return expect_and_parse(idx, args, [&verbosity](const char* val) {
    verbosity = parse_verbosity_level(val);
  });
}

static auto
parse_delay(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  auto& delay_ms = opts.delay_ms;
  return expect_and_parse(idx, args, [&delay_ms](const char* val) {
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
  const bool parsed = expect_and_parse(idx, args, [&opts](const char* val) {
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

  return true;
}

static auto
parse_address(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  if (idx + 1 >= args.size()) {
    return false;
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
  return expect_and_parse(idx, args, [&max_batch_size](const char* val) {
    const int tmp = std::stoi(val);
    if (tmp <= 0) {
      throw std::invalid_argument("Must be > 0.");
    }
    max_batch_size = tmp;
  });
}

static auto
parse_metrics_port(RuntimeConfig& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  auto& metrics_port = opts.metrics_port;
  return expect_and_parse(idx, args, [&metrics_port](const char* val) {
    metrics_port = std::stoi(val);
    if (metrics_port <= 0) {
      throw std::invalid_argument("Must be > 0.");
    }
  });
}

static auto
parse_scheduler(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  if (idx + 1 >= args.size()) {
    return false;
  }
  ++idx;
  opts.scheduler = args[idx];
  return true;
}

static auto
parse_pregen_inputs(RuntimeConfig& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  auto& pregen = opts.pregen_inputs;
  return expect_and_parse(idx, args, [&pregen](const char* val) {
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
  return expect_and_parse(idx, args, [&warmup](const char* val) {
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
  return expect_and_parse(idx, args, [&seed](const char* val) {
    const auto tmp = std::stoll(val);
    if (tmp < 0) {
      throw std::invalid_argument("Must be >= 0.");
    }
    seed = static_cast<uint64_t>(tmp);
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
          {"--pregen-inputs", parse_pregen_inputs},
          {"--warmup-iterations", parse_warmup_iterations},
          {"--seed", parse_seed},
      };

  for (size_t idx = 1; idx < args_span.size(); ++idx) {
    const std::string_view arg = args_span[idx];

    if (arg == "--sync") {
      opts.synchronous = true;
    } else if (arg == "--no_cpu") {
      opts.use_cpu = false;
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
  check_required(!opts.model_path.empty(), "--model", missing);
  const bool have_shapes = std::any_of(
      opts.inputs.begin(), opts.inputs.end(),
      [](const auto& t) { return !t.dims.empty(); });
  const bool have_types = std::all_of(
      opts.inputs.begin(), opts.inputs.end(),
      [](const auto& t) { return t.type != at::ScalarType::Undefined; });
  check_required(have_shapes, "--shape or --shapes", missing);
  check_required(have_types, "--types", missing);

  if (have_shapes && have_types) {
    for (const auto& t : opts.inputs) {
      if (t.dims.empty() || t.type == at::ScalarType::Undefined) {
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
      opts.max_message_bytes = compute_max_message_bytes(
          opts.max_batch_size, opts.inputs, opts.outputs,
          opts.max_message_bytes);
    }
  }

  return opts;
}

}  // namespace starpu_server
