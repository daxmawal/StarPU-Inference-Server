#pragma once

#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <format>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "config_loader_helpers.hpp"
#include "datatype_utils.hpp"

namespace starpu_server { inline namespace config_loader_detail {

void
parse_tensor_name(
    const YAML::Node& node, std::size_t index, std::string_view label,
    TensorConfig& tensor_config)
{
  if (const YAML::Node name_node = node["name"]) {
    const auto name_path = make_indexed_field_path(label, index, "name");
    if (!name_node.IsScalar()) {
      throw std::invalid_argument(
          std::format("{} must be a scalar string", name_path));
    }
    tensor_config.name = name_node.as<std::string>();
  }
}

auto
parse_tensor_dims(
    const YAML::Node& node, std::size_t index, std::string_view label,
    std::size_t max_dims) -> std::vector<int64_t>
{
  const YAML::Node dims_node = node["dims"];
  const auto dims_path = make_indexed_field_path(label, index, "dims");
  if (!dims_node) {
    throw std::invalid_argument(std::format("{} is required", dims_path));
  }
  if (!dims_node.IsSequence()) {
    throw std::invalid_argument(
        std::format("{} must be a sequence of integers", dims_path));
  }
  if (dims_node.size() > max_dims) {
    std::ostringstream oss;
    oss << dims_path << " must have at most " << max_dims << " entries";
    throw std::invalid_argument(oss.str());
  }

  std::vector<int64_t> dims;
  dims.reserve(dims_node.size());
  for (std::size_t j = 0; j < dims_node.size(); ++j) {
    const auto& dim_node = dims_node[j];
    const auto dim_path = std::format("{}[{}]", dims_path, j);
    const auto dim_value =
        parse_scalar<long long>(dim_node, dim_path, "an integer");

    if (dim_value <= 0) {
      throw std::invalid_argument(std::format("{} must be > 0", dim_path));
    }
    if (dim_value > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::format(
          "{} must be <= {}", dim_path, std::numeric_limits<int>::max()));
    }
    dims.push_back(static_cast<int64_t>(dim_value));
  }
  return dims;
}

auto
parse_tensor_type(
    const YAML::Node& node, std::size_t index,
    std::string_view label) -> at::ScalarType
{
  const YAML::Node type_node = node["data_type"];
  const auto type_path = make_indexed_field_path(label, index, "data_type");
  if (!type_node) {
    throw std::invalid_argument(std::format("{} is required", type_path));
  }
  if (!type_node.IsScalar()) {
    throw std::invalid_argument(
        std::format("{} must be a scalar string", type_path));
  }
  try {
    return string_to_scalar_type(type_node.as<std::string>());
  }
  catch (const std::invalid_argument& exception) {
    throw std::invalid_argument(
        std::format("{}: {}", type_path, exception.what()));
  }
}

auto
parse_tensor_entry(
    const YAML::Node& node, std::size_t index, std::string_view label,
    std::size_t max_dims) -> TensorConfig
{
  const auto entry_path = make_indexed_path(label, index);
  if (!node.IsMap()) {
    throw std::invalid_argument(
        std::format("{} must be a mapping", entry_path));
  }

  TensorConfig tensor_config{};
  parse_tensor_name(node, index, label, tensor_config);
  tensor_config.dims = parse_tensor_dims(node, index, label, max_dims);
  tensor_config.type = parse_tensor_type(node, index, label);
  return tensor_config;
}

auto
parse_tensor_nodes(
    const YAML::Node& nodes, std::size_t max_inputs, std::string_view label,
    std::size_t max_dims) -> std::vector<TensorConfig>
{
  std::vector<TensorConfig> tensors;
  if (!nodes) {
    return tensors;
  }
  if (!nodes.IsSequence()) {
    throw std::invalid_argument(
        std::format("{} must be a sequence of tensor definitions", label));
  }
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto& node = nodes[i];
    if (tensors.size() >= max_inputs) {
      std::ostringstream oss;
      oss << label << " must have at most " << max_inputs << " entries";
      throw std::invalid_argument(oss.str());
    }
    tensors.push_back(parse_tensor_entry(node, i, label, max_dims));
  }
  return tensors;
}

void
parse_io_nodes(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["inputs"]) {
    auto& model = ensure_model(cfg);
    model.inputs = parse_tensor_nodes(
        root["inputs"], cfg.limits.max_inputs, "inputs", cfg.limits.max_dims);
  }
  if (root["outputs"]) {
    auto& model = ensure_model(cfg);
    model.outputs = parse_tensor_nodes(
        root["outputs"], cfg.limits.max_inputs, "outputs", cfg.limits.max_dims);
  }
}

}}  // namespace starpu_server::config_loader_detail
