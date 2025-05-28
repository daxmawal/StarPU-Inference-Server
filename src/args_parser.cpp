#include "args_parser.hpp"

#include <functional>
#include <iostream>
#include <sstream>
#include <unordered_map>

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
std::vector<int64_t>
parse_shape_string(const std::string& shape_str)
{
  std::vector<int64_t> shape;
  std::stringstream ss(shape_str);
  std::string item;

  while (std::getline(ss, item, 'x')) {
    try {
      shape.push_back(std::stoll(item));
    }
    catch (const std::exception& e) {
      throw std::invalid_argument(
          "Shape contains non-integer: " + std::string(e.what()));
    }
  }

  if (shape.empty())
    throw std::invalid_argument("Shape string is empty or invalid.");

  return shape;
}

std::vector<std::vector<int64_t>>
parse_shapes_string(const std::string& shapes_str)
{
  std::vector<std::vector<int64_t>> shapes;
  std::stringstream ss(shapes_str);
  std::string shape_str;

  while (std::getline(ss, shape_str, ',')) {
    shapes.push_back(parse_shape_string(shape_str));
  }

  if (shapes.empty())
    throw std::invalid_argument("No valid shapes provided.");

  return shapes;
}

at::ScalarType
parse_type_string(const std::string& type_str)
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

  auto it = type_map.find(type_str);
  if (it == type_map.end())
    throw std::invalid_argument("Unsupported type: " + type_str);
  return it->second;
}

std::vector<at::ScalarType>
parse_types_string(const std::string& types_str)
{
  std::vector<at::ScalarType> types;
  std::stringstream ss(types_str);
  std::string type_str;

  while (std::getline(ss, type_str, ',')) {
    types.push_back(parse_type_string(type_str));
  }

  return types;
}

VerbosityLevel
parse_verbosity_level(const std::string& val)
{
  int level = std::stoi(val);
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
bool
try_parse(const std::string& argname, const char* value, Func&& func)
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
ProgramOptions
parse_arguments(int argc, char* argv[])
{
  ProgramOptions opts;

  std::unordered_map<std::string, std::function<bool(int&, char**)>> dispatch =
      {{"--scheduler",
        [&](int& i, char** args) {
          if (i + 1 >= argc)
            return false;
          opts.scheduler = args[++i];
          return true;
        }},
       {"--model",
        [&](int& i, char** args) {
          if (i + 1 >= argc)
            return false;
          opts.model_path = args[++i];
          return true;
        }},
       {"--iterations",
        [&](int& i, char** args) {
          if (i + 1 >= argc)
            return false;
          return try_parse("iterations", args[++i], [&](const char* val) {
            int tmp = std::stoi(val);
            if (tmp <= 0)
              throw std::invalid_argument("Must be > 0.");
            opts.iterations = static_cast<unsigned int>(tmp);
          });
        }},
       {"--shape",
        [&](int& i, char** args) {
          if (i + 1 >= argc)
            return false;
          return try_parse("shape", args[++i], [&](const char* val) {
            opts.input_shapes = {parse_shape_string(val)};
          });
        }},
       {"--shapes",
        [&](int& i, char** args) {
          if (i + 1 >= argc)
            return false;
          return try_parse("shapes", args[++i], [&](const char* val) {
            opts.input_shapes = parse_shapes_string(val);
          });
        }},
       {"--types",
        [&](int& i, char** args) {
          if (i + 1 >= argc)
            return false;
          return try_parse("types", args[++i], [&](const char* val) {
            opts.input_types = parse_types_string(val);
          });
        }},
       {"--verbose",
        [&](int& i, char** args) {
          if (i + 1 >= argc)
            return false;
          return try_parse("verbose", args[++i], [&](const char* val) {
            opts.verbosity = parse_verbosity_level(val);
          });
        }},
       {"--delay",
        [&](int& i, char** args) {
          if (i + 1 >= argc)
            return false;
          return try_parse("delay", args[++i], [&](const char* val) {
            opts.delay_ms = std::stoi(val);
            if (opts.delay_ms < 0)
              throw std::invalid_argument("Must be >= 0.");
          });
        }},
       {"--device-ids", [&](int& i, char** args) {
          if (i + 1 >= argc)
            return false;
          return try_parse("device-ids", args[++i], [&](const char* val) {
            opts.use_cuda = true;
            std::stringstream ss(val);
            std::string id_str;
            while (std::getline(ss, id_str, ',')) {
              int id = std::stoi(id_str);
              if (id < 0)
                throw std::invalid_argument("Must be >= 0.");
              opts.device_ids.push_back(static_cast<unsigned int>(id));
            }
            if (opts.device_ids.empty())
              throw std::invalid_argument("No device IDs provided.");
          });
        }}};

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--sync") {
      opts.synchronous = true;
    } else if (arg == "--no_cpu") {
      opts.use_cpu = false;
    } else if (arg == "--help") {
      opts.show_help = true;
      return opts;
    } else if (auto it = dispatch.find(arg); it != dispatch.end()) {
      if (!it->second(i, argv)) {
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
  if (opts.model_path.empty()) {
    log_error("--model option is required.");
    opts.valid = false;
  }

  if (!opts.input_types.empty() &&
      opts.input_types.size() != opts.input_shapes.size()) {
    log_error("Number of --types must match number of input shapes.");
    opts.valid = false;
  }

  if (opts.input_shapes.empty() || opts.input_types.empty()) {
    log_error("Both --shape/--shapes and --types must be provided.");
    opts.valid = false;
  }

  return opts;
}