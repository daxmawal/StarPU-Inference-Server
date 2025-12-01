#include "trace_plotting.hpp"

#include <cstdlib>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <vector>

#include "utils/batching_trace_logger.hpp"
#include "utils/logger.hpp"
#include "utils/runtime_config.hpp"

namespace starpu_server {

namespace {

auto
shell_quote(const std::string& value) -> std::string
{
  std::string quoted;
  quoted.reserve(value.size() + 2);
  quoted.push_back('\'');
  for (char character : value) {
    if (character == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(character);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

auto
candidate_plot_scripts(const RuntimeConfig& opts)
    -> std::vector<std::filesystem::path>
{
  std::vector<std::filesystem::path> candidates;
  candidates.emplace_back("scripts/plot_batch_summary.py");
  if (!opts.config_path.empty()) {
    const auto config_dir =
        std::filesystem::path(opts.config_path).parent_path();
    candidates.emplace_back(config_dir / "scripts/plot_batch_summary.py");
  }
  std::error_code exe_ec;
  const auto exe_path = std::filesystem::read_symlink("/proc/self/exe", exe_ec);
  if (!exe_ec) {
    candidates.emplace_back(
        std::filesystem::path(exe_path).parent_path() /
        "../scripts/plot_batch_summary.py");
  }
  return candidates;
}

auto
locate_plot_script(const RuntimeConfig& opts)
    -> std::optional<std::filesystem::path>
{
  for (const auto& candidate : candidate_plot_scripts(opts)) {
    if (candidate.empty()) {
      continue;
    }
    auto resolved = candidate;
    if (!resolved.is_absolute()) {
      std::error_code abs_ec;
      const auto absolute = std::filesystem::absolute(resolved, abs_ec);
      if (!abs_ec) {
        resolved = absolute;
      }
    }
    std::error_code exists_ec;
    if (std::filesystem::exists(resolved, exists_ec) && !exists_ec) {
      return resolved;
    }
  }
  return std::nullopt;
}

auto
plots_output_path(const std::filesystem::path& summary_path)
    -> std::filesystem::path
{
  auto filename = summary_path.stem().string();
  if (const auto pos = filename.rfind("_summary"); pos != std::string::npos) {
    filename.erase(pos);
  }
  filename += "_plots.png";
  auto output = summary_path;
  output.replace_filename(filename);
  return output;
}

}  // namespace

void
run_trace_plots_if_enabled(const RuntimeConfig& opts)
{
  if (!opts.batching.trace_enabled) {
    return;
  }

  const auto& tracer = BatchingTraceLogger::instance();
  std::vector<std::filesystem::path> summary_paths;
  if (const auto main_summary = tracer.summary_file_path()) {
    summary_paths.push_back(*main_summary);
  }
  if (summary_paths.empty()) {
    log_warning(
        "Tracing was enabled but no batching_trace_summary.csv files were "
        "produced; skipping plot generation.");
    return;
  }

  const auto script_path = locate_plot_script(opts);
  if (!script_path) {
    log_warning(
        "Unable to locate scripts/plot_batch_summary.py; skipping plot "
        "generation.");
    return;
  }

  for (const auto& summary_path : summary_paths) {
    if (std::error_code err_code;
        !std::filesystem::exists(summary_path, err_code) || err_code) {
      log_warning(std::format(
          "Tracing summary file '{}' not found; skipping plot generation.",
          summary_path.string()));
      continue;
    }
    const auto output_path = plots_output_path(summary_path);
    const std::string command = std::format(
        "python3 {} {} --output {}", shell_quote(script_path->string()),
        shell_quote(summary_path.string()), shell_quote(output_path.string()));
    const int return_code = std::system(command.c_str());
    if (return_code != 0) {
      log_warning(std::format(
          "Failed to generate batching latency plots for '{}'; command '{}' "
          "exited with code {}.",
          summary_path.string(), command, return_code));
    } else {
      log_info(
          opts.verbosity,
          std::format(
              "Batching latency plots written to '{}'.", output_path.string()));
    }
  }
}

}  // namespace starpu_server
