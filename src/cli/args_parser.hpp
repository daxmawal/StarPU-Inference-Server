#pragma once
#include <ATen/core/ScalarType.h>

#include <span>
#include <string>
#include <vector>

#include "logger.hpp"
#include "runtime_config.hpp"

namespace starpu_server {

inline auto
get_help_message(const char* prog_name) -> std::string
{
  std::string msg = "Usage: ";
  msg += prog_name;
  msg +=
      " [OPTIONS]\n"
      "\nOptions:\n"
      "  --scheduler [name]      Scheduler type (default: lws)\n"
      "  --config [file]         YAML configuration file\n"
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
      "  --metrics-port [port]   Port for metrics exposition (default: 9090)\n"
      "  --max-batch-size N      Maximum inference batch size\n"
      "  --pregen-inputs N       Number of pregenerated inputs (default: 10)\n"
      "  --warmup-iterations N   Warmup iterations per CUDA device "
      "(default:2)\n"
      "  --verbose [0-4]         Verbosity level: 0=silent to 4=trace\n"
      "  --help                  Show this help message\n";
  return msg;
}

inline void
display_help(const char* prog_name)
{
  log_info(VerbosityLevel::Info, get_help_message(prog_name));
}

auto parse_arguments(std::span<char*> args_span, RuntimeConfig opts = {})
    -> RuntimeConfig;
}  // namespace starpu_server
