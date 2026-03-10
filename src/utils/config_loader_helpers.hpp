// Internal helpers shared across config_loader_*.hpp implementation files.
// This header is an implementation detail of config_loader.cpp only.
#pragma once

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#include "config_loader.hpp"

namespace starpu_server { namespace {

constexpr int kMinPort = 1;
constexpr int kMaxPort = 65535;
const std::filesystem::path kDefaultTraceOutputFile{
    RuntimeConfig::BatchingSettings{}.trace_output_path};

template <typename T>
auto
parse_scalar(
    const YAML::Node& node, std::string_view key,
    std::string_view type_desc) -> T
{
  if (!node.IsScalar()) {
    throw std::invalid_argument(std::format("{} must be {}", key, type_desc));
  }
  try {
    return node.as<T>();
  }
  catch (const YAML::BadConversion&) {
    throw std::invalid_argument(std::format("{} must be {}", key, type_desc));
  }
}

auto
ensure_model(RuntimeConfig& cfg) -> ModelConfig&
{
  if (!cfg.model.has_value()) {
    cfg.model = ModelConfig{};
  }
  return *cfg.model;
}

auto
make_indexed_path(std::string_view base, std::size_t index) -> std::string
{
  return std::format("{}[{}]", base, index);
}

auto
make_indexed_field_path(
    std::string_view base, std::size_t index,
    std::string_view field) -> std::string
{
  return std::format("{}[{}].{}", base, index, field);
}

auto
resolve_trace_output_directory(std::string directory) -> std::string
{
  if (directory.empty()) {
    return directory;
  }

  std::filesystem::path directory_path(directory);
  std::error_code status_ec;
  const bool exists = std::filesystem::exists(directory_path, status_ec);
  if (status_ec) {
    throw std::filesystem::filesystem_error(
        "trace_output", directory_path, status_ec);
  }

  if (exists) {
    std::error_code type_ec;
    if (!std::filesystem::is_directory(directory_path, type_ec)) {
      throw std::invalid_argument("trace_output must be a directory path");
    }
  } else {
    const auto extension = directory_path.extension();
    if (!extension.empty() &&
        extension == kDefaultTraceOutputFile.extension()) {
      throw std::invalid_argument(
          "trace_output must be a directory path (omit the filename)");
    }
  }

  return (directory_path / kDefaultTraceOutputFile).string();
}

}}  // namespace starpu_server
