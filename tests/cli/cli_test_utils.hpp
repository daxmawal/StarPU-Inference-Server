#pragma once

#include <gtest/gtest.h>

#include <initializer_list>
#include <vector>

#include "cli/args_parser.hpp"
#include "utils/runtime_config.hpp"

inline constexpr const char kCliHelpMessage[] =
    "Usage: Inference Engine [OPTIONS]\n"
    "\nOptions:\n"
    "  --scheduler [name]      Scheduler type (default: lws)\n"
    "  --model [path]          Path to TorchScript model file (.pt)\n"
    "  --iterations [num]      Number of iterations (default: 1)\n"
    "  --shape 1x3x224x224     Shape of a single input tensor\n"
    "  --shapes shape1,shape2  Shapes for multiple input tensors\n"
    "  --types float,int       Input tensor types (default: float)\n"
    "  --sync                  Run tasks in synchronous mode\n"
    "  --delay [ms]            Delay between jobs (default: 0)\n"
    "  --no_cpu                Disable CPU usage\n"
    "  --device-ids 0,1        GPU device IDs for inference\n"
    "  --address ADDR          gRPC server listen address\n"
    "  --max-msg-size BYTES    Max gRPC message size in bytes\n"
    "  --verbose [0-4]         Verbosity level: 0=silent to 4=trace\n"
    "  --help                  Show this help message\n";

inline std::vector<char*>
build_argv(std::initializer_list<const char*> args)
{
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (const char* arg : args) {
    argv.push_back(const_cast<char*>(arg));
  }
  return argv;
}

inline auto
build_valid_cli_args() -> std::vector<char*>
{
  return build_argv(
      {"program", "--model", "model.pt", "--shape", "1x1", "--types", "float"});
}

inline auto
parse(std::initializer_list<const char*> args) -> starpu_server::RuntimeConfig
{
  auto argv = build_argv(args);
  return starpu_server::parse_arguments({argv.data(), argv.size()});
}

inline void
expect_invalid(std::initializer_list<const char*> args)
{
  EXPECT_FALSE(parse(args).valid);
}