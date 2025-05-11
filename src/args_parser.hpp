#pragma once

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
  bool synchronous = false;
  bool valid = true;
};

void display_help(const char* prog_name);
std::vector<int64_t> parse_shape_string(const std::string& shape_str);
ProgramOptions parse_arguments(int argc, char* argv[]);