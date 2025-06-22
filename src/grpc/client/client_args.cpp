#include "client_args.hpp"

#include <functional>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
parse_type_string(const std::string& type_str) -> at::ScalarType
{
  static const std::unordered_map<std::string, at::ScalarType> type_map = {
      {"float", at::kFloat},       {"float32", at::kFloat},
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
scalar_type_to_string(at::ScalarType type) -> std::string
{
  switch (type) {
    case at::kFloat:
      return "FP32";
    case at::kDouble:
      return "FP64";
    case at::kHalf:
      return "FP16";
    case at::kBFloat16:
      return "BF16";
    case at::kInt:
      return "INT32";
    case at::kLong:
      return "INT64";
    case at::kShort:
      return "INT16";
    case at::kChar:
      return "INT8";
    case at::kByte:
      return "UINT8";
    case at::kBool:
      return "BOOL";
    default:
      return "FP32";
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
  catch (const std::exception& e) {
    std::cerr << "Invalid value: " << e.what() << '\n';
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
  return try_parse(args[++idx], std::forward<Func>(parser));
}

// =============================================================================
// Individual Argument Parsers for --model, --shape, etc.
// =============================================================================

auto
parse_iterations(ClientConfig& cfg, size_t& idx, std::span<const char*> args)
    -> bool
{
  return expect_and_parse(idx, args, [&](const char* val) {
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
  return expect_and_parse(idx, args, [&](const char* val) {
    cfg.delay_ms = std::stoi(val);
    if (cfg.delay_ms < 0) {
      throw std::invalid_argument("Must be >= 0.");
    }
  });
}

auto
parse_shape(ClientConfig& cfg, size_t& idx, std::span<const char*> args) -> bool
{
  return expect_and_parse(
      idx, args, [&](const char* val) { cfg.shape = parse_shape_string(val); });
}

auto
parse_type(ClientConfig& cfg, size_t& idx, std::span<const char*> args) -> bool
{
  return expect_and_parse(
      idx, args, [&](const char* val) { cfg.type = parse_type_string(val); });
}

auto
parse_server(ClientConfig& cfg, size_t& idx, std::span<const char*> args)
    -> bool
{
  return expect_and_parse(
      idx, args, [&](const char* val) { cfg.server_address = val; });
}

auto
parse_model(ClientConfig& cfg, size_t& idx, std::span<const char*> args) -> bool
{
  return expect_and_parse(
      idx, args, [&](const char* val) { cfg.model_name = val; });
}

auto
parse_version(ClientConfig& cfg, size_t& idx, std::span<const char*> args)
    -> bool
{
  return expect_and_parse(
      idx, args, [&](const char* val) { cfg.model_version = val; });
}

// =============================================================================
// Dispatch Argument Parser (Main parser loop)
// =============================================================================

auto
parse_argument_values(std::span<const char*> args_span, ClientConfig& cfg)
    -> bool
{
  static const std::unordered_map<std::string, std::function<bool(size_t&)>>
      dispatch = {
          {"--iterations",
           [&](size_t& idx) { return parse_iterations(cfg, idx, args_span); }},
          {"--delay",
           [&](size_t& idx) { return parse_delay(cfg, idx, args_span); }},
          {"--shape",
           [&](size_t& idx) { return parse_shape(cfg, idx, args_span); }},
          {"--type",
           [&](size_t& idx) { return parse_type(cfg, idx, args_span); }},
          {"--server",
           [&](size_t& idx) { return parse_server(cfg, idx, args_span); }},
          {"--model",
           [&](size_t& idx) { return parse_model(cfg, idx, args_span); }},
          {"--version",
           [&](size_t& idx) { return parse_version(cfg, idx, args_span); }},
      };

  for (size_t idx = 1; idx < args_span.size(); ++idx) {
    const std::string arg = args_span[idx];

    if (arg == "--help" || arg == "-h") {
      cfg.show_help = true;
      return true;
    }

    auto iter = dispatch.find(arg);
    if (iter != dispatch.end()) {
      if (!iter->second(idx)) {
        return false;
      }
      continue;
    }

    std::cerr << "Unknown argument: " << arg
              << ". Use --help to see valid options.\n";
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
      std::cerr << opt << " option is required.\n";
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