#pragma once
#include <ATen/core/ScalarType.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "utils/datatype_utils.hpp"
#include "utils/logger.hpp"

namespace starpu_server {
struct InputConfig {
  std::string name;
  std::vector<int64_t> shape;
  at::ScalarType type = at::kFloat;
};

struct ClientConfig {
  std::vector<int64_t> shape;
  at::ScalarType type = at::kFloat;
  std::vector<InputConfig> inputs;
  std::string server_address = "localhost:50051";
  std::string model_name = "example";
  std::string model_version = "1";
  int request_nb = 1;
  int delay_us = 0;
  VerbosityLevel verbosity = VerbosityLevel::Info;
  bool show_help = false;
  bool valid = true;
};

void display_client_help(const char* prog_name);
auto parse_client_args(std::span<const char*> args) -> ClientConfig;
}  // namespace starpu_server
