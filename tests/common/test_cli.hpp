#pragma once

#include <gtest/gtest.h>

#include <concepts>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <ranges>
#include <span>
#include <vector>

#include "cli/args_parser.hpp"
#include "utils/runtime_config.hpp"

struct OwnedArgv {
  std::vector<std::string> storage;
  std::vector<char*> argv;
  auto data() -> char** { return argv.data(); }
  [[nodiscard]] auto size() const -> std::size_t { return argv.size(); }
};

template <std::ranges::input_range Range>
  requires std::convertible_to<
               std::ranges::range_reference_t<Range>, const char*>
inline auto
build_argv(Range&& args) -> OwnedArgv
{
  OwnedArgv owned;
  if constexpr (std::ranges::sized_range<Range>) {
    const auto size = static_cast<std::size_t>(std::ranges::size(args));
    owned.storage.reserve(size);
    owned.argv.reserve(size);
  }
  for (const char* arg : args) {
    owned.storage.emplace_back(arg);
    owned.argv.push_back(owned.storage.back().data());
  }
  return owned;
}

inline auto
build_argv(std::initializer_list<const char*> args) -> OwnedArgv
{
  return build_argv(std::span<const char* const>(args.begin(), args.size()));
}

inline auto
create_empty_file(const std::string& name) -> std::string
{
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream(path).close();
  return path.string();
}

inline auto
test_model_path() -> const std::string&
{
  static const std::string path = create_empty_file("model.pt");
  return path;
}

inline auto
test_config_path() -> const std::string&
{
  static const std::string path = create_empty_file("config.yaml");
  return path;
}

inline auto
build_valid_cli_args() -> OwnedArgv
{
  return build_argv(
      {"program", "--model", test_model_path().c_str(), "--shape", "1x1",
       "--types", "float"});
}

inline auto
parse(std::span<const char* const> args) -> starpu_server::RuntimeConfig
{
  auto argv = build_argv(args);
  return starpu_server::parse_arguments({argv.data(), argv.size()});
}

inline auto
parse(std::initializer_list<const char*> args) -> starpu_server::RuntimeConfig
{
  return parse(std::span<const char* const>(args.begin(), args.size()));
}

inline void
expect_invalid(std::span<const char* const> args)
{
  EXPECT_FALSE(parse(args).valid);
}

inline void
expect_invalid(std::initializer_list<const char*> args)
{
  expect_invalid(std::span<const char* const>(args.begin(), args.size()));
}
