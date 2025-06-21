#include "client_args.hpp"

#include <functional>
#include <iostream>
#include <span>
#include <sstream>
#include <unordered_map>

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
  std::cout << "Usage: " << prog_name << " [OPTIONS]\n"
            << "  --iterations N    Number of requests to send (default: 1)\n"
            << "  --delay MS        Delay between requests in ms (default: 0)\n"
            << "  --shape WxHxC     Input tensor shape (e.g., 1x3x224x224)\n"
            << "  --type TYPE       Input tensor type (e.g., float32)\n"
            << "  --help            Show this help message\n";
}

auto
parse_client_args(const std::span<const char*> args) -> ClientConfig
{
  ClientConfig cfg;

  using Handler = std::function<void(size_t&)>;
  std::unordered_map<std::string, Handler> handlers;

  handlers["--iterations"] = [&](size_t& iteration) {
    if (iteration + 1 >= args.size()) {
      throw std::invalid_argument("Missing value for --iterations");
    }
    cfg.iterations = std::stoi(args[++iteration]);
    if (cfg.iterations <= 0) {
      throw std::invalid_argument("--iterations must be > 0");
    }
  };

  handlers["--delay"] = [&](size_t& iteration) {
    if (iteration + 1 >= args.size()) {
      throw std::invalid_argument("Missing value for --delay");
    }
    cfg.delay_ms = std::stoi(args[++iteration]);
    if (cfg.delay_ms < 0) {
      throw std::invalid_argument("--delay must be >= 0");
    }
  };

  handlers["--shape"] = [&](size_t& iteration) {
    if (iteration + 1 >= args.size()) {
      throw std::invalid_argument("Missing value for --shape");
    }
    cfg.shape = parse_shape_string(args[++iteration]);
  };

  handlers["--type"] = [&](size_t& iteration) {
    if (iteration + 1 >= args.size()) {
      throw std::invalid_argument("Missing value for --type");
    }
    cfg.type = parse_type_string(args[++iteration]);
  };

  handlers["--help"] = [&](size_t&) { cfg.show_help = true; };

  handlers["-h"] = handlers["--help"];  // alias

  for (size_t idx = 1; idx < args.size(); ++idx) {
    const std::string arg = args[idx];
    try {
      if (auto iterator = handlers.find(arg); iterator != handlers.end()) {
        iterator->second(idx);
      } else {
        std::cerr << "Unknown option: " << arg << "\n";
        cfg.valid = false;
      }
    }
    catch (const std::exception& e) {
      std::cerr << e.what() << "\n";
      cfg.valid = false;
    }
  }

  if (!cfg.show_help && cfg.shape.empty()) {
    std::cerr << "--shape is required\n";
    cfg.valid = false;
  }

  return cfg;
}
