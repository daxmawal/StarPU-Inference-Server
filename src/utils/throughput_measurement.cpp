#include "throughput_measurement.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "logger.hpp"

namespace starpu_server {

namespace {

auto
parse_json_value(std::string_view value_str) -> std::string
{
  auto trimmed = value_str;
  while (!trimmed.empty() && std::isspace(trimmed.front())) {
    trimmed.remove_prefix(1);
  }
  while (!trimmed.empty() && std::isspace(trimmed.back())) {
    trimmed.remove_suffix(1);
  }
  return std::string(trimmed);
}

auto
parse_json_file(std::string_view content)
    -> std::optional<ThroughputMeasurement>
{
  ThroughputMeasurement result;

  size_t sig_pos = content.find("\"config_signature\"");
  if (sig_pos == std::string::npos) {
    return std::nullopt;
  }
  size_t sig_start = content.find('\"', sig_pos + 20);
  if (sig_start == std::string::npos) {
    return std::nullopt;
  }
  size_t sig_end = content.find('\"', sig_start + 1);
  if (sig_end == std::string::npos) {
    return std::nullopt;
  }
  result.config_signature =
      std::string(content.substr(sig_start + 1, sig_end - sig_start - 1));

  size_t gpu_pos = content.find("\"throughput_gpu\"");
  if (gpu_pos != std::string::npos) {
    size_t gpu_colon = content.find(':', gpu_pos);
    if (gpu_colon != std::string::npos) {
      size_t gpu_start = gpu_colon + 1;
      size_t gpu_end = content.find(',', gpu_start);
      if (gpu_end == std::string::npos) {
        gpu_end = content.find('}', gpu_start);
      }
      if (gpu_end != std::string::npos) {
        auto gpu_str =
            parse_json_value(content.substr(gpu_start, gpu_end - gpu_start));
        try {
          result.throughput_gpu = std::stod(gpu_str);
        }
        catch (...) {
        }
      }
    }
  }

  size_t cpu_pos = content.find("\"throughput_cpu\"");
  if (cpu_pos != std::string::npos) {
    size_t cpu_colon = content.find(':', cpu_pos);
    if (cpu_colon != std::string::npos) {
      size_t cpu_start = cpu_colon + 1;
      size_t cpu_end = content.find(',', cpu_start);
      if (cpu_end == std::string::npos) {
        cpu_end = content.find('}', cpu_start);
      }
      if (cpu_end != std::string::npos) {
        auto cpu_str =
            parse_json_value(content.substr(cpu_start, cpu_end - cpu_start));
        try {
          result.throughput_cpu = std::stod(cpu_str);
        }
        catch (...) {
        }
      }
    }
  }

  return result;
}

auto
parse_old_text_format(std::string_view content)
    -> std::optional<ThroughputMeasurement>
{
  std::string content_str(content);
  std::istringstream iss(content_str);
  std::string signature;
  double throughput = -1.0;

  if (std::getline(iss, signature) && !signature.empty()) {
    if (iss >> throughput && throughput > 0.0) {
      ThroughputMeasurement result;
      result.config_signature = signature;
      result.throughput_gpu = throughput;
      return result;
    }
  }
  return std::nullopt;
}

}  // namespace

auto
load_cached_throughput_measurements(const std::filesystem::path& file_path)
    -> std::optional<ThroughputMeasurement>
{
  std::error_code ec;
  if (!std::filesystem::exists(file_path, ec) || ec) {
    return std::nullopt;
  }

  try {
    std::ifstream file(file_path);
    if (!file) {
      return std::nullopt;
    }
    std::string content(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    if (content.empty()) {
      return std::nullopt;
    }

    if (auto result = parse_json_file(content)) {
      return result;
    }

    if (auto result = parse_old_text_format(content)) {
      return result;
    }

    return std::nullopt;
  }
  catch (...) {
    return std::nullopt;
  }
}

auto
save_throughput_measurements(
    const std::filesystem::path& file_path,
    const ThroughputMeasurement& measurements) -> bool
{
  try {
    file_path.parent_path().empty() ||
        std::filesystem::create_directories(file_path.parent_path());

    std::ofstream out(file_path, std::ios::trunc);
    if (!out) {
      return false;
    }

    out << "{\n";
    out << "  \"config_signature\": \"" << measurements.config_signature
        << "\",\n";

    if (measurements.throughput_gpu >= 0.0) {
      out << "  \"throughput_gpu\": " << std::fixed << std::setprecision(6)
          << measurements.throughput_gpu << ",\n";
    }

    if (measurements.throughput_cpu >= 0.0) {
      out << "  \"throughput_cpu\": " << std::fixed << std::setprecision(6)
          << measurements.throughput_cpu << "\n";
    } else if (measurements.throughput_gpu >= 0.0) {
      out.seekp(-2, std::ios_base::end);
      out << "\n";
    }

    out << "}\n";
    return out.good();
  }
  catch (const std::exception& e) {
    log_warning(
        std::string("Failed to save throughput measurements: ") + e.what());
    return false;
  }
}

}  // namespace starpu_server
