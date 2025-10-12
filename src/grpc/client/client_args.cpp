#include "client_args.hpp"

#include <format>
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
    int64_t dim{};
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
      throw std::invalid_argument(
          "Shape contains non-positive dimension: " + item);
    }
    shape.push_back(dim);
  }
  if (shape.empty()) {
    throw std::invalid_argument("Shape string is empty or invalid.");
  }
  return shape;
}

}  // namespace

void
display_client_help(const char* prog_name)
{
  std::cout
      << "Usage: " << prog_name << " [OPTIONS]\n"
      << "  --request-number N Number of requests to send (default: 1)\n"
      << "  --delay US        Delay between requests in microseconds (default: "
         "0)\n"
      << "  --shape WxHxC     Input tensor shape (e.g., 1x3x224x224)\n"
      << "  --type TYPE       Input tensor type (e.g., float32)\n"
      << "  --input NAME:SHAPE:TYPE  Specify an input (may be repeated)\n"
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
parse_request_nb(ClientConfig& cfg, size_t& idx, std::span<const char*> args)
    -> bool
{
  return expect_and_parse(idx, args, [&cfg](const char* val) {
    const int tmp = std::stoi(val);
    if (tmp <= 0) {
      throw std::invalid_argument("Must be > 0.");
    }
    cfg.request_nb = tmp;
  });
}

auto
parse_delay(ClientConfig& cfg, size_t& idx, std::span<const char*> args) -> bool
{
  return expect_and_parse(idx, args, [&cfg](const char* val) {
    cfg.delay_us = std::stoi(val);
    if (cfg.delay_us < 0) {
      throw std::invalid_argument("Must be >= 0.");
    }
  });
}

auto
parse_shape(ClientConfig& cfg, size_t& idx, std::span<const char*> args) -> bool
{
  return expect_and_parse(idx, args, [&cfg](const char* val) {
    cfg.shape = parse_shape_string(val);
    if (cfg.inputs.empty()) {
      cfg.inputs.push_back({"input", cfg.shape, cfg.type});
    } else {
      cfg.inputs[0].shape = cfg.shape;
    }
  });
}


auto
parse_type(ClientConfig& cfg, size_t& idx, std::span<const char*> args) -> bool
{
  return expect_and_parse(idx, args, [&cfg](const char* val) {
    cfg.type = string_to_scalar_type(val);
    if (cfg.inputs.empty()) {
      cfg.inputs.push_back({"input", cfg.shape, cfg.type});
    } else {
      cfg.inputs[0].type = cfg.type;
    }
  });
}

auto
parse_input(ClientConfig& cfg, size_t& idx, std::span<const char*> args) -> bool
{
  return expect_and_parse(idx, args, [&cfg](const char* val) {
    std::string token(val);
    std::stringstream token_stream(token);
    std::string name;
    std::string shape_str;
    std::string type_str;
    if (!std::getline(token_stream, name, ':') ||
        !std::getline(token_stream, shape_str, ':') ||
        !std::getline(token_stream, type_str)) {
      throw std::invalid_argument("Input must be NAME:SHAPE:TYPE");
    }
    InputConfig input{};
    input.name = name;
    input.shape = parse_shape_string(shape_str);
    input.type = string_to_scalar_type(type_str);
    cfg.inputs.push_back(std::move(input));
    if (cfg.inputs.size() == 1) {
      cfg.shape = cfg.inputs[0].shape;
      cfg.type = cfg.inputs[0].type;
    }
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
      std::string, bool (*)(ClientConfig&, size_t&, std::span<const char*>),
      TransparentHash, std::equal_to<>>
      dispatch = {
          {"--request-number", parse_request_nb},
          {"--delay", parse_delay},
          {"--shape", parse_shape},
          {"--type", parse_type},
          {"--input", parse_input},
          {"--server", parse_server},
          {"--model", parse_model},
          {"--version", parse_version},
          {"--verbose", parse_verbose},
      };

  for (size_t idx = 1; idx < args_span.size(); ++idx) {
    const std::string arg = args_span[idx];

    if (arg == "--help" || arg == "-h") {
      cfg.show_help = true;
      return true;
    }


    if (auto iter = dispatch.find(arg); iter != dispatch.end()) {
      if (!iter->second(cfg, idx, args_span)) {
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
  check_required(!cfg.inputs.empty(), "--input", missing);
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
