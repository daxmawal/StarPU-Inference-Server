#pragma once

#include <gtest/gtest.h>

#include <initializer_list>
#include <vector>

#include "cli/args_parser.hpp"
#include "utils/runtime_config.hpp"

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

inline std::vector<char*>
build_argv(const std::vector<const char*>& args)
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

inline auto
parse(const std::vector<const char*>& args) -> starpu_server::RuntimeConfig
{
  auto argv = build_argv(args);
  return starpu_server::parse_arguments({argv.data(), argv.size()});
}

inline void
expect_invalid(std::initializer_list<const char*> args)
{
  EXPECT_FALSE(parse(args).valid);
}

inline void
expect_invalid(const std::vector<const char*>& args)
{
  EXPECT_FALSE(parse(args).valid);
}
