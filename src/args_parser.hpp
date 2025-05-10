#ifndef ARGS_PARSER_HPP
#define ARGS_PARSER_HPP

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>

struct ProgramOptions 
{
  std::string scheduler = "lws";
  std::string model_path;
  int iterations = 1;
  std::vector<int64_t> input_shape;
  bool show_help = false;
  bool valid = true;
};

void display_help(const char* prog_name) 
{
  std::cout << "Usage: " << prog_name << " [OPTIONS]\n"
            << "\nOptions:\n"
            << "  --scheduler [name]      Scheduler type (default: lws)\n"
            << "  --model [path]          Path to TorchScript model file (.pt)\n"
            << "  --iterations [num]      Number of iterations (default: 1)\n"
            << "  --shape 1x3x224x224     Shape of input tensor (e.g., for image models)\n"
            << "  --help                  Show this help message\n";
}

std::vector<int64_t> parse_shape_string(const std::string& shape_str) 
{
  std::vector<int64_t> shape;
  std::stringstream ss(shape_str);
  std::string item;

  while (std::getline(ss, item, 'x')) 
  {
    try 
    {
      shape.push_back(std::stoll(item));
    } 
    catch (...) 
    {
      throw std::invalid_argument("Shape contains non-integer values.");
    }
  }

  if (shape.empty()) 
    throw std::invalid_argument("Shape string is empty or invalid.");

  return shape;
}

ProgramOptions parse_arguments(int argc, char* argv[]) 
{
  ProgramOptions opts;

  for (int i = 1; i < argc; ++i) 
  {
    std::string arg = argv[i];

    if (arg == "--scheduler" && i + 1 < argc) 
    {
      opts.scheduler = argv[++i];
    }
    else if (arg == "--model" && i + 1 < argc)
    {
      opts.model_path = argv[++i];
    }
    else if (arg == "--iterations" && i + 1 < argc) 
    {
      try 
      {
        opts.iterations = std::stoi(argv[++i]);
        if (opts.iterations <= 0) 
          throw std::invalid_argument("Iterations must be positive.");
      }
      catch (const std::exception& e) 
      {
        std::cerr << "Invalid value for iterations: " << e.what() << "\n";
        opts.valid = false;
        return opts;
      }
    }
    else if (arg == "--shape" && i + 1 < argc)
    {
      try 
      {
        opts.input_shape = parse_shape_string(argv[++i]);
      }
      catch (const std::exception& e) 
      {
        std::cerr << "Invalid shape: " << e.what() << "\n";
        opts.valid = false;
        return opts;
      }
    }
    else if (arg == "--help") 
    {
      opts.show_help = true;
      return opts;
    }
    else 
    {
      std::cerr << "Unknown argument: " << arg << "\n";
      opts.valid = false;
      return opts;
    }
  }

  if (opts.model_path.empty()) 
  {
    std::cerr << "Error: --model option is required.\n";
    opts.valid = false;
  }

  return opts;
}

#endif // ARGS_PARSER_HPP
