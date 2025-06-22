#pragma once
#include <ATen/core/ScalarType.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "utils/logger.hpp"

struct ClientConfig {
  int iterations = 1;
  int delay_ms = 0;
  std::vector<int64_t> shape;
  at::ScalarType type = at::kFloat;
  std::string server_address = "localhost:50051";
  std::string model_name = "example";
  std::string model_version = "1";
  VerbosityLevel verbosity = VerbosityLevel::Info;
  bool show_help = false;
  bool valid = true;
};

void display_client_help(const char* prog_name);
auto parse_client_args(const std::span<const char*> args) -> ClientConfig;
auto scalar_type_to_string(at::ScalarType type) -> std::string;