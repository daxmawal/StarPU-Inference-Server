#include "args_parser.hpp"

#include <c10/core/ScalarType.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "logger.hpp"
#include "runtime_config.hpp"
namespace starpu_server {

// =============================================================================
// Shape and Type Parsing: Handle --shape, --shapes, and --types arguments
// =============================================================================

auto
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
    catch (const std::exception& e) {
      throw std::invalid_argument(
          "Shape contains non-integer: " + std::string(e.what()));
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

auto
parse_shapes_string(const std::string& shapes_str)
    -> std::vector<std::vector<int64_t>>
{
  std::vector<std::vector<int64_t>> shapes;
  std::stringstream shape_stream(shapes_str);
  std::string shape_str;

  while (std::getline(shape_stream, shape_str, ',')) {
    shapes.push_back(parse_shape_string(shape_str));
  }

  if (shapes.empty()) {
    throw std::invalid_argument("No valid shapes provided.");
  }

  return shapes;
}

struct TransparentHash {
  using is_transparent = void;
  std::size_t operator()(std::string_view key) const noexcept
  {
    return std::hash<std::string_view>{}(key);
  }
};

auto
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

  auto iterator = type_map.find(std::string_view(type_str));
  if (iterator == type_map.end()) {
    throw std::invalid_argument("Unsupported type: " + type_str);
  }
  return iterator->second;
}

auto
parse_types_string(const std::string& types_str) -> std::vector<at::ScalarType>
{
  std::vector<at::ScalarType> types;
  std::stringstream shape_stream(types_str);
  std::string type_str;

  while (std::getline(shape_stream, type_str, ',')) {
    types.push_back(parse_type_string(type_str));
  }

  return types;
}

// =============================================================================
// Verbosity Parsing
// =============================================================================

auto
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

void
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
  catch (const std::exception& e) {
    log_error(std::string("Invalid value: ") + e.what());
    return false;
  }
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

auto
parse_model(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  if (idx + 1 >= args.size()) {
    return false;
  }
  ++idx;
  opts.model_path = args[idx];
  return true;
}

auto
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

auto
parse_shape(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  auto& input_shapes = opts.input_shapes;
  return expect_and_parse(idx, args, [&input_shapes](const char* val) {
    input_shapes = {parse_shape_string(val)};
  });
}

auto
parse_shapes(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  auto& input_shapes = opts.input_shapes;
  return expect_and_parse(idx, args, [&input_shapes](const char* val) {
    input_shapes = parse_shapes_string(val);
  });
}

auto
parse_types(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  auto& input_types = opts.input_types;
  return expect_and_parse(idx, args, [&input_types](const char* val) {
    input_types = parse_types_string(val);
  });
}

auto
parse_verbose(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  auto& verbosity = opts.verbosity;
  return expect_and_parse(idx, args, [&verbosity](const char* val) {
    verbosity = parse_verbosity_level(val);
  });
}

auto
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

auto
parse_device_ids(RuntimeConfig& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  return expect_and_parse(idx, args, [&opts](const char* val) {
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
}

auto
parse_address(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  if (idx + 1 >= args.size()) {
    return false;
  }
  ++idx;
  opts.server_address = args[idx];
  return true;
}

auto
parse_max_msg_size(RuntimeConfig& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  auto& max_message_bytes = opts.max_message_bytes;
  return expect_and_parse(idx, args, [&max_message_bytes](const char* val) {
    const int tmp = std::stoi(val);
    if (tmp <= 0) {
      throw std::invalid_argument("Must be > 0.");
    }
    max_message_bytes = tmp;
  });
}

auto
parse_scheduler(RuntimeConfig& opts, size_t& idx, std::span<char*> args) -> bool
{
  if (idx + 1 >= args.size()) {
    return false;
  }
  ++idx;
  opts.scheduler = args[idx];
  return true;
}

// =============================================================================
// Dispatch Argument Parser (Main parser loop)
// =============================================================================

auto
parse_argument_values(std::span<char*> args_span, RuntimeConfig& opts) -> bool
{
  static const std::unordered_map<std::string, std::function<bool(size_t&)>>
      dispatch = {
          {"--model",
           [&opts, &args_span](size_t& idx) {
             return parse_model(opts, idx, args_span);
           }},
          {"--iterations",
           [&opts, &args_span](size_t& idx) {
             return parse_iterations(opts, idx, args_span);
           }},
          {"--shape",
           [&opts, &args_span](size_t& idx) {
             return parse_shape(opts, idx, args_span);
           }},
          {"--shapes",
           [&opts, &args_span](size_t& idx) {
             return parse_shapes(opts, idx, args_span);
           }},
          {"--types",
           [&opts, &args_span](size_t& idx) {
             return parse_types(opts, idx, args_span);
           }},
          {"--verbose",
           [&opts, &args_span](size_t& idx) {
             return parse_verbose(opts, idx, args_span);
           }},
          {"--delay",
           [&opts, &args_span](size_t& idx) {
             return parse_delay(opts, idx, args_span);
           }},
          {"--device-ids",
           [&opts, &args_span](size_t& idx) {
             return parse_device_ids(opts, idx, args_span);
           }},
          {"--scheduler",
           [&opts, &args_span](size_t& idx) {
             return parse_scheduler(opts, idx, args_span);
           }},
          {"--address",
           [&opts, &args_span](size_t& idx) {
             return parse_address(opts, idx, args_span);
           }},
          {"--max-msg-size",
           [&opts, &args_span](size_t& idx) {
             return parse_max_msg_size(opts, idx, args_span);
           }},
      };

  for (size_t idx = 1; idx < args_span.size(); ++idx) {
    const std::string arg = args_span[idx];

    if (arg == "--sync") {
      opts.synchronous = true;
    } else if (arg == "--no_cpu") {
      opts.use_cpu = false;
    } else if (arg == "--help" || arg == "-h") {
      opts.show_help = true;
      return true;
    } else if (auto iter = dispatch.find(arg); iter != dispatch.end()) {
      if (!iter->second(idx)) {
        return false;
      }
    } else {
      log_error(
          "Unknown argument: " + arg + ". Use --help to see valid options.");
      return false;
    }
  }

  return true;
}

// =============================================================================
// Config Validation: Ensures all required fields are present and consistent
// =============================================================================

auto
validate_config(RuntimeConfig& opts) -> void
{
  std::vector<std::string> missing;
  check_required(!opts.model_path.empty(), "--model", missing);
  check_required(!opts.input_shapes.empty(), "--shape or --shapes", missing);
  check_required(!opts.input_types.empty(), "--types", missing);

  if (opts.input_shapes.size() != opts.input_types.size()) {
    log_error("Number of --types must match number of input shapes.");
    opts.valid = false;
  }

  if (!missing.empty()) {
    for (const auto& opt : missing) {
      log_error(opt + " option is required.");
    }
    opts.valid = false;
  }
}

// =============================================================================
// Top-Level Entry: Parses all arguments into a RuntimeConfig object
// =============================================================================

auto
parse_arguments(std::span<char*> args_span) -> RuntimeConfig
{
  RuntimeConfig opts;

  if (!parse_argument_values(args_span, opts)) {
    opts.valid = false;
    return opts;
  }

  if (!opts.show_help) {
    validate_config(opts);
  }

  return opts;
}

}  // namespace starpu_server