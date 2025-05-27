#include "args_parser.hpp"

#include <functional>
#include <unordered_map>

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
      << "  --shapes shape1,shape2  Shapes for multiple input tensors, e.g. "
         "1x3x224x224,1x10\n"
      << "  --types float,int       Types for input tensors, e.g. float,int "
         "(default: float)\n"
      << "  --sync                  Run tasks in synchronous mode (default: "
         "async)\n"
      << "  --delay [ms]            Delay in milliseconds between inference "
         "jobs (default: 0)\n"
      << "  --no_cpu                Disable CPU utilisation by the scheduler\n "
      << " --device-ids            Cuda device id, where to perform inference\n"
      << "  --verbose [0-4]         Verbosity level: 0=silent, 1=info "
         "(default), 2=debug, 3=trace\n"
      << "  --help                  Show this help message\n";
}

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
          std::string("Shape contains non-integer values: ") + e.what());
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
    throw std::invalid_argument("No valid shapes were provided.");

  return shapes;
}

at::ScalarType
parse_type_string(const std::string& type_str)
{
  if (type_str == "float" || type_str == "float32")
    return at::kFloat;
  if (type_str == "double" || type_str == "float64")
    return at::kDouble;
  if (type_str == "half" || type_str == "float16")
    return at::kHalf;
  if (type_str == "bfloat16")
    return at::kBFloat16;

  if (type_str == "int" || type_str == "int32")
    return at::kInt;
  if (type_str == "long" || type_str == "int64")
    return at::kLong;
  if (type_str == "short" || type_str == "int16")
    return at::kShort;
  if (type_str == "char" || type_str == "int8")
    return at::kChar;

  if (type_str == "byte" || type_str == "uint8")
    return at::kByte;

  if (type_str == "bool")
    return at::kBool;
  if (type_str == "complex64")
    return at::kComplexFloat;
  if (type_str == "complex128")
    return at::kComplexDouble;

  throw std::invalid_argument("Unsupported type: " + type_str);
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
              throw std::invalid_argument("Iterations must be positive.");
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
              throw std::invalid_argument("Delay must be non-negative.");
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
                throw std::invalid_argument("Device ID must be non-negative.");
              opts.device_ids.push_back(static_cast<unsigned int>(id));
            }
            if (opts.device_ids.empty()) {
              throw std::invalid_argument("No device IDs provided.");
            }
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

  if (opts.model_path.empty()) {
    log_error("--model option is required.\n");
    opts.valid = false;
  }

  if (!opts.input_types.empty() &&
      opts.input_types.size() != opts.input_shapes.size()) {
    log_error("Number of types must match number of input shapes.\n");
    opts.valid = false;
  }

  if (opts.input_shapes.empty() || opts.input_types.empty()) {
    log_error("Error: both --shape/--shapes and --types must be provided.\n");
    opts.valid = false;
  }

  return opts;
}