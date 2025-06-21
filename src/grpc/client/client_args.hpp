#pragma once
#include <ATen/core/ScalarType.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

struct ClientConfig {
  int iterations = 1;
  int delay_ms = 0;
  std::vector<int64_t> shape;
  at::ScalarType type = at::kFloat;
  bool show_help = false;
  bool valid = true;
};

void display_client_help(const char* prog_name);
auto parse_client_args(const std::span<const char*> args) -> ClientConfig;
auto scalar_type_to_string(at::ScalarType type) -> std::string;