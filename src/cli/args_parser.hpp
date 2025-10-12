#pragma once
#include <span>
#include <string>

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
      "  --request-number [num]   Number of requests (default: 1)\n"
      "  --shape 1x3x224x224     Shape of a single input tensor\n"
      "  --shapes shape1,shape2  Shapes for multiple input tensors\n"
      "  --types float,int       Input tensor types (default: float)\n"
      "  --input name:DIMS:TYPE  Combined input spec (repeatable);\n"
      "                          e.g., input0:32x3x224x224:float32\n"
      "  --sync                  Run tasks in synchronous mode\n"
      "  --delay [us]            Delay between jobs in microseconds (default: "
      "0)\n"
      "  --no_cpu                Disable CPU usage\n"
      "  --device-ids 0,1        GPU device IDs for inference\n"
      "  --address ADDR          gRPC server listen address\n"
      "  --metrics-port [port]   Port for metrics exposition (default: 9090)\n"
      "  --max-batch-size N      Maximum inference batch size\n"
      "  --input-slots N         Number of reusable input slots\n"
      "  --slots N               Alias for --input-slots\n"
      "  --pregen-inputs N       Number of pregenerated inputs (default: 10)\n"
      "  --warmup-request_nb N   Warmup request_nb per CUDA device "
      "(default:2)\n"
      "  --rtol [value]          Relative tolerance for validation (default: "
      "1e-3)\n"
      "  --atol [value]          Absolute tolerance for validation (default: "
      "1e-5)\n"
      "  --no-validate           Disable inference result validation\n"
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
