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

// =============================================================================
// Display help
// =============================================================================
void
display_help(const char* prog_name)
{
  std::cout
      << "Usage: " << prog_name << " [OPTIONS]\n"
      << "\nOptions:\n"
      << "  --scheduler [name]      Scheduler type (default: lws)\n"
      << "  --model [path]          Path to TorchScript model file (.pt)\n"
      << "  --iterations [num]      Number of iterations (default: 1)\n"
      << "  --shape 1x3x224x224     Shape of a single input tensor\n"
      << "  --shapes shape1,shape2  Shapes for multiple input tensors\n"
      << "  --types float,int       Input tensor types (default: float)\n"
      << "  --sync                  Run tasks in synchronous mode\n"
      << "  --delay [ms]            Delay between jobs (default: 0)\n"
      << "  --no_cpu                Disable CPU usage\n"
      << "  --device-ids 0,1        GPU device IDs for inference\n"
      << "  --verbose [0-4]         Verbosity level: 0=silent to 4=trace\n"
      << "  --help                  Show this help message\n";
}

// =============================================================================
// Parsing utilities
// =============================================================================
auto
parse_shape_string(const std::string& shape_str) -> std::vector<int64_t>
{
  std::vector<int64_t> shape;
  std::stringstream shape_stream(shape_str);
  std::string item;

  while (std::getline(shape_stream, item, 'x')) {
    try {
      shape.push_back(std::stoll(item));
    }
    catch (const std::exception& e) {
      throw std::invalid_argument(
          "Shape contains non-integer: " + std::string(e.what()));
    }
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

auto
parse_type_string(const std::string& type_str) -> at::ScalarType
{
  static const std::unordered_map<std::string, at::ScalarType> type_map = {
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
      {"complex128", at::kComplexDouble}};

  auto iterator = type_map.find(type_str);
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

auto
parse_verbosity_level(const std::string& val) -> VerbosityLevel
{
  const int level = std::stoi(val);
  switch (level) {
    case 0:
      return VerbosityLevel::Silent;
    case 1:
      return VerbosityLevel::Info;
    case 2:
      return VerbosityLevel::Stats;
    case 3:
      return VerbosityLevel::Debug;
    case 4:
      return VerbosityLevel::Trace;
    default:
      throw std::invalid_argument("Invalid verbosity level: " + val);
  }
}

template <typename Func>
auto
try_parse(const std::string& argname, const char* value, const Func&& func)
    -> bool
{
  try {
    func(value);
    return true;
  }
  catch (const std::exception& e) {
    log_error("Invalid value for " + argname + ": " + e.what());
    return false;
  }
}

// =============================================================================
// Argument parser
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


// -------------------- Generic helpers --------------------

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
  return try_parse(args[++idx], std::forward<Func>(parser));
}

// -------------------- Individual parsers --------------------

namespace parsers {

auto
parse_model(ProgramOptions& opts, size_t& idx, std::span<char*> args) -> bool
{
  if (idx + 1 >= args.size()) {
    return false;
  }
  ++idx;
  opts.model_path = args[idx];
  return true;
}

auto
parse_iterations(ProgramOptions& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  return expect_and_parse(idx, args, [&](const char* val) {
    const int tmp = std::stoi(val);
    if (tmp <= 0) {
      throw std::invalid_argument("Must be > 0.");
    }
    opts.iterations = static_cast<unsigned int>(tmp);
  });
}

auto
parse_shape(ProgramOptions& opts, size_t& idx, std::span<char*> args) -> bool
{
  return expect_and_parse(idx, args, [&](const char* val) {
    opts.input_shapes = {parse_shape_string(val)};
  });
}

auto
parse_shapes(ProgramOptions& opts, size_t& idx, std::span<char*> args) -> bool
{
  return expect_and_parse(idx, args, [&](const char* val) {
    opts.input_shapes = parse_shapes_string(val);
  });
}

auto
parse_types(ProgramOptions& opts, size_t& idx, std::span<char*> args) -> bool
{
  return expect_and_parse(idx, args, [&](const char* val) {
    opts.input_types = parse_types_string(val);
  });
}

auto
parse_verbose(ProgramOptions& opts, size_t& idx, std::span<char*> args) -> bool
{
  return expect_and_parse(idx, args, [&](const char* val) {
    opts.verbosity = parse_verbosity_level(val);
  });
}

auto
parse_delay(ProgramOptions& opts, size_t& idx, std::span<char*> args) -> bool
{
  return expect_and_parse(idx, args, [&](const char* val) {
    opts.delay_ms = std::stoi(val);
    if (opts.delay_ms < 0) {
      throw std::invalid_argument("Must be >= 0.");
    }
  });
}

auto
parse_device_ids(ProgramOptions& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  return expect_and_parse(idx, args, [&](const char* val) {
    opts.use_cuda = true;
    std::stringstream shape_stream(val);
    std::string id_str;
    while (std::getline(shape_stream, id_str, ',')) {
      int device_id = std::stoi(id_str);
      if (device_id < 0) {
        throw std::invalid_argument("Must be >= 0.");
      }
      opts.device_ids.push_back(static_cast<unsigned int>(device_id));
    }
    if (opts.device_ids.empty()) {
      throw std::invalid_argument("No device IDs provided.");
    }
  });
}

auto
parse_scheduler(ProgramOptions& opts, size_t& idx, std::span<char*> args)
    -> bool
{
  if (idx + 1 >= args.size()) {
    return false;
  }
  opts.scheduler = args[++idx];
  return true;
}
}  // namespace parsers

// -------------------- Principal function --------------------

auto
parse_arguments(std::span<char*> args_span) -> ProgramOptions
{
  ProgramOptions opts;

  static const std::unordered_map<std::string, std::function<bool(size_t&)>>
      dispatch = {
          {"--model",
           [&](size_t& idx) {
             return parsers::parse_model(opts, idx, args_span);
           }},
          {"--iterations",
           [&](size_t& idx) {
             return parsers::parse_iterations(opts, idx, args_span);
           }},
          {"--shape",
           [&](size_t& idx) {
             return parsers::parse_shape(opts, idx, args_span);
           }},
          {"--shapes",
           [&](size_t& idx) {
             return parsers::parse_shapes(opts, idx, args_span);
           }},
          {"--types",
           [&](size_t& idx) {
             return parsers::parse_types(opts, idx, args_span);
           }},
          {"--verbose",
           [&](size_t& idx) {
             return parsers::parse_verbose(opts, idx, args_span);
           }},
          {"--delay",
           [&](size_t& idx) {
             return parsers::parse_delay(opts, idx, args_span);
           }},
          {"--device-ids",
           [&](size_t& idx) {
             return parsers::parse_device_ids(opts, idx, args_span);
           }},
          {"--scheduler",
           [&](size_t& idx) {
             return parsers::parse_scheduler(opts, idx, args_span);
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
      return opts;
    } else if (auto iterator = dispatch.find(arg); iterator != dispatch.end()) {
      if (!iterator->second(idx)) {
        opts.valid = false;
        return opts;
      }
    } else {
      log_error(
          "Unknown argument: " + arg + ". Use --help to see valid options.");
      opts.valid = false;
      return opts;
    }
  }

  // Post-validation
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

  return opts;
}