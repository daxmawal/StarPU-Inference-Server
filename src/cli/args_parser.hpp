#pragma once
#include <ATen/core/ScalarType.h>

#include <span>
#include <string>
#include <vector>

#include "logger.hpp"
#include "runtime_config.hpp"

namespace starpu_server {
inline void
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
      << "  --address ADDR          gRPC server listen address\n"
      << "  --max-msg-size BYTES    Max gRPC message size in bytes\n"
      << "  --verbose [0-4]         Verbosity level: 0=silent to 4=trace\n"
      << "  --help                  Show this help message\n";
}

auto parse_arguments(std::span<char*> args_span) -> RuntimeConfig;
}  // namespace starpu_server
