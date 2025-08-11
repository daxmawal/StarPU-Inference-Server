#pragma once

#include <gtest/gtest.h>

#include <initializer_list>
#include <vector>

#include "cli/args_parser.hpp"
#include "utils/runtime_config.hpp"

struct OwnedArgv {
  std::vector<std::string> storage;
  std::vector<char*> argv;
  auto data() -> char** { return argv.data(); }
  [[nodiscard]] auto size() const -> std::size_t { return argv.size(); }
};

inline auto
build_argv(std::initializer_list<const char*> args) -> OwnedArgv
{
  OwnedArgv owned;
  owned.storage.reserve(args.size());
  owned.argv.reserve(args.size());
  for (const char* arg : args) {
    owned.storage.emplace_back(arg);
    owned.argv.push_back(owned.storage.back().data());
  }
  return owned;
}

inline auto
build_argv(const std::vector<const char*>& args) -> OwnedArgv
{
  OwnedArgv owned;
  owned.storage.reserve(args.size());
  owned.argv.reserve(args.size());
  for (const char* arg : args) {
    owned.storage.emplace_back(arg);
    owned.argv.push_back(owned.storage.back().data());
  }
  return owned;
}

inline auto
build_valid_cli_args() -> OwnedArgv
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
