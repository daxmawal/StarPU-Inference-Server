#ifndef ARGS_PARSER_HPP
#define ARGS_PARSER_HPP

#include <iostream>
#include <string>
#include <stdexcept>

struct ProgramOptions 
{
  std::string scheduler = "lws";
  std::string model_path;
  int iterations = 1;
  bool show_help = false;
  bool valid = true;
};

void display_help(const char* prog_name) 
{
  std::cout << "Usage: " << prog_name << " [OPTIONS]\n"
            << "\nOptions:\n"
            << "  --scheduler [name]    Scheduler type (default: lws)\n"
            << "  --model [path]        Path to TorchScript model file (.pt)\n"
            << "  --iterations [num]    Number of iterations (default: 1)\n"
            << "  --help                Show this help message\n";
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
        {
          throw std::invalid_argument("Iterations must be positive.");
        }
      }
      catch (const std::exception& e) 
      {
        std::cerr << "Invalid value for iterations: " << e.what() << "\n";
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
