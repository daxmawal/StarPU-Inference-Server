#include "client_args.hpp"

#include <format>
#include <functional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/datatype_utils.hpp"
#include "utils/logger.hpp"
#include "utils/transparent_hash.hpp"

namespace starpu_server {
namespace {

auto
parse_shape_string(const std::string& shape_str) -> std::vector<int64_t>
{
  std::vector<int64_t> shape;
  std::stringstream string_stream(shape_str);
  std::string item;
  while (std::getline(string_stream, item, 'x')) {
    try {
      shape.push_back(std::stoll(item));
    }
    catch (const std::invalid_argument& e) {
      throw std::invalid_argument(
          "Shape contains non-integer: " + std::string(e.what()));
    }
    catch (const std::out_of_range& e) {
      throw std::out_of_range(
          "Shape value out of range: " + std::string(e.what()));
    }
  }
  if (shape.empty()) {
    throw std::invalid_argument("Shape string is empty or invalid.");
  }
  return shape;
}

auto
parse_type_string(const std::string& type_str) -> at::ScalarType
{
  static const std::unordered_map<
      std::string, at::ScalarType, TransparentHash, std::equal_to<>>
      type_map = {{"float", at::kFloat},       {"float32", at::kFloat},
                  {"double", at::kDouble},     {"float64", at::kDouble},
                  {"half", at::kHalf},         {"float16", at::kHalf},
                  {"bfloat16", at::kBFloat16}, {"int", at::kInt},
                  {"int32", at::kInt},         {"long", at::kLong},
                  {"int64", at::kLong},        {"short", at::kShort},
                  {"int16", at::kShort},       {"char", at::kChar},
                  {"int8", at::kChar},         {"byte", at::kByte},
                  {"uint8", at::kByte},        {"bool", at::kBool}};
  auto iterator = type_map.find(type_str);
  if (iterator == type_map.end()) {
    throw std::invalid_argument("Unsupported type: " + type_str);
  }
  return iterator->second;
}

}  // namespace

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

void
display_client_help(const char* prog_name)
{
  std::cout
      << "Usage: " << prog_name << " [OPTIONS]\n"
      << "  --iterations N    Number of requests to send (default: 1)\n"
      << "  --delay MS        Delay between requests in ms (default: 0)\n"
      << "  --shape WxHxC     Input tensor shape (e.g., 1x3x224x224)\n"
      << "  --type TYPE       Input tensor type (e.g., float32)\n"
      << "  --server ADDR     gRPC server address (default: localhost:50051)\n"
      << "  --model NAME      Model name (default: example)\n"
      << "  --version VER     Model version (default: 1)\n"
      << "  --verbose [0-4]   Verbosity level: 0=silent to 4=trace\n"
      << "  --help            Show this help message\n";
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
  catch (const std::invalid_argument& e) {
    log_error(std::format("Invalid value: {}", e.what()));
    return false;
  }
  catch (const std::out_of_range& e) {
    log_error(std::format("Value out of range: {}", e.what()));
    return false;
  }
}

template <typename Func>
auto
expect_and_parse(size_t& idx, std::span<const char*> args, Func&& parser)
    -> bool
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
parse_iterations(ClientConfig& cfg, size_t& idx, std::span<const char*> args)
    -> bool
{
  return expect_and_parse(idx, args, [&cfg](const char* val) {
    const int tmp = std::stoi(val);
    if (tmp <= 0) {
      throw std::invalid_argument("Must be > 0.");
    }
    cfg.iterations = tmp;
  });
}

auto
parse_delay(ClientConfig& cfg, size_t& idx, std::span<const char*> args) -> bool
{
  return expect_and_parse(idx, args, [&cfg](const char* val) {
    cfg.delay_ms = std::stoi(val);
    if (cfg.delay_ms < 0) {
      throw std::invalid_argument("Must be >= 0.");
    }
  });
}

auto
parse_shape(ClientConfig& cfg, size_t& idx, std::span<const char*> args) -> bool
{
  return expect_and_parse(idx, args, [&cfg](const char* val) {
    cfg.shape = parse_shape_string(val);
  });
}


auto
parse_type(ClientConfig& cfg, size_t& idx, std::span<const char*> args) -> bool
{
  return expect_and_parse(idx, args, [&cfg](const char* val) {
    cfg.type = parse_type_string(val);
  });
}

auto
parse_server(ClientConfig& cfg, size_t& idx, std::span<const char*> args)
    -> bool
{
  return expect_and_parse(
      idx, args, [&cfg](const char* val) { cfg.server_address = val; });
}

auto
parse_model(ClientConfig& cfg, size_t& idx, std::span<const char*> args) -> bool
{
  return expect_and_parse(
      idx, args, [&cfg](const char* val) { cfg.model_name = val; });
}

auto
parse_version(ClientConfig& cfg, size_t& idx, std::span<const char*> args)
    -> bool
{
  return expect_and_parse(
      idx, args, [&cfg](const char* val) { cfg.model_version = val; });
}

auto
parse_verbose(ClientConfig& cfg, size_t& idx, std::span<const char*> args)
    -> bool
{
  return expect_and_parse(idx, args, [&cfg](const char* val) {
    cfg.verbosity = parse_verbosity_level(val);
  });
}

// =============================================================================
// Dispatch Argument Parser (Main parser loop)
// =============================================================================

auto
parse_argument_values(std::span<const char*> args_span, ClientConfig& cfg)
    -> bool
{
  static const std::unordered_map<
      std::string, std::function<bool(size_t&)>, TransparentHash,
      std::equal_to<>>
      dispatch = {
          {"--iterations",
           [&cfg, &args_span](size_t& idx) {
             return parse_iterations(cfg, idx, args_span);
           }},
          {"--delay",
           [&cfg, &args_span](size_t& idx) {
             return parse_delay(cfg, idx, args_span);
           }},
          {"--shape",
           [&cfg, &args_span](size_t& idx) {
             return parse_shape(cfg, idx, args_span);
           }},
          {"--type",
           [&cfg, &args_span](size_t& idx) {
             return parse_type(cfg, idx, args_span);
           }},
          {"--server",
           [&cfg, &args_span](size_t& idx) {
             return parse_server(cfg, idx, args_span);
           }},
          {"--model",
           [&cfg, &args_span](size_t& idx) {
             return parse_model(cfg, idx, args_span);
           }},
          {"--version",
           [&cfg, &args_span](size_t& idx) {
             return parse_version(cfg, idx, args_span);
           }},
          {"--verbose",
           [&cfg, &args_span](size_t& idx) {
             return parse_verbose(cfg, idx, args_span);
           }},
      };

  for (size_t idx = 1; idx < args_span.size(); ++idx) {
    const std::string arg = args_span[idx];

    if (arg == "--help" || arg == "-h") {
      cfg.show_help = true;
      return true;
    }


    if (auto iter = dispatch.find(arg); iter != dispatch.end()) {
      if (!iter->second(idx)) {
        return false;
      }
      continue;
    }

    log_error(std::format(
        "Unknown argument: {}. Use --help to see valid options.", arg));
    return false;
  }

  return true;
}

// =============================================================================
// Config Validation: Ensures all required fields are present and consistent
// =============================================================================

auto
validate_config(ClientConfig& cfg) -> void
{
  std::vector<std::string> missing;
  check_required(!cfg.shape.empty(), "--shape", missing);
  if (!missing.empty()) {
    for (const auto& opt : missing) {
      log_error(std::format("{} option is required.", opt));
    }
    cfg.valid = false;
  }
}

// =============================================================================
// Top-Level Entry: Parses all arguments into a RuntimeConfig object
// =============================================================================

auto
parse_client_args(const std::span<const char*> args) -> ClientConfig
{
  ClientConfig cfg;

  if (!parse_argument_values(args, cfg)) {
    cfg.valid = false;
    return cfg;
  }

  if (!cfg.show_help) {
    validate_config(cfg);
  }

  return cfg;
}
}  // namespace starpu_server