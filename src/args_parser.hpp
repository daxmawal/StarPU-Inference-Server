#pragma once

#include <ATen/core/ScalarType.h>

#include <string>
#include <vector>

struct ProgramOptions {
  std::string scheduler = "lws";
  std::string model_path;
  unsigned int iterations = 1;
  bool synchronous = false;
  int delay_ms = 0;
  bool show_help = false;
  bool valid = true;
  bool use_cpu = true;
  std::vector<unsigned int> device_ids;
  bool use_cuda = false;

  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<at::ScalarType> input_types;
};

void display_help(const char* prog_name);
std::vector<int64_t> parse_shape_string(const std::string& shape_str);
ProgramOptions parse_arguments(int argc, char* argv[]);