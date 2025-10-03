#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace starpu_server {
// Logging utilities
// -----------------
// Writes to stdout/stderr are guarded by a global mutex so
// that concurrent logging from multiple threads does not
// interleave. Keep the lock scope minimal for better
// concurrency.

inline std::mutex log_mutex;

enum class VerbosityLevel : std::uint8_t {
  Silent = 0,
  Info = 1,
  Stats = 2,
  Debug = 3,
  Trace = 4
};

// =============================================================================
// Utility: parse verbosity level from string or number
// =============================================================================

inline auto
parse_verbosity_level(const std::string& val) -> VerbosityLevel
{
  using enum VerbosityLevel;

  const auto first = val.find_first_not_of(" \t\n\r\f\v");
  const auto last = val.find_last_not_of(" \t\n\r\f\v");
  const std::string trimmed = first == std::string::npos
                                  ? std::string{}
                                  : val.substr(first, last - first + 1);

  if (std::all_of(trimmed.begin(), trimmed.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
      })) {
    int level{};
    try {
      level = std::stoi(trimmed);
    }
    catch (const std::invalid_argument&) {
      throw std::invalid_argument("Invalid verbosity level: " + trimmed);
    }
    catch (const std::out_of_range&) {
      throw std::invalid_argument("Verbosity level out of range: " + trimmed);
    }
    switch (level) {
      case 0:
        return Silent;
      case 1:
        return Info;
      case 2:
        return Stats;
      case 3:
        return Debug;
      case 4:
        return Trace;
      default:
        throw std::invalid_argument("Invalid verbosity level: " + trimmed);
    }
  }

  std::string lower(trimmed.size(), '\0');
  std::transform(
      trimmed.begin(), trimmed.end(), lower.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lower == "silent") {
    return Silent;
  }
  if (lower == "info") {
    return Info;
  }
  if (lower == "stats") {
    return Stats;
  }
  if (lower == "debug") {
    return Debug;
  }
  if (lower == "trace") {
    return Trace;
  }
  throw std::invalid_argument("Invalid verbosity level: " + trimmed);
}

// =============================================================================
// Utility: color and label mapping for verbosity levels
// =============================================================================

inline auto
verbosity_style(const VerbosityLevel level)
    -> std::pair<const char*, const char*>
{
  using enum VerbosityLevel;
  switch (level) {
    case Info:
      return {"\x1b[1;32m", "[INFO] "};  // Green
    case Stats:
      return {"\x1b[1;35m", "[STATS] "};  // Magenta
    case Debug:
      return {"\x1b[1;34m", "[DEBUG] "};  // Blue
    case Trace:
      return {"\x1b[1;90m", "[TRACE] "};  // Gray
    default:
      return {"", ""};
  }
}

// =============================================================================
// Verbosity-controlled logging
// =============================================================================

inline void
log_verbose(
    const VerbosityLevel level, const VerbosityLevel current_level,
    const std::string& message)
{
  if (std::to_underlying(current_level) >= std::to_underlying(level)) {
    auto [color, label] = verbosity_style(level);
    const std::scoped_lock lock(log_mutex);
    std::cout << color << label << message << "\x1b[0m\n" << std::flush;
  }
}

// =============================================================================
// Shortcut wrappers
// =============================================================================

inline void
log_info(const VerbosityLevel lvl, const std::string& msg)
{
  log_verbose(VerbosityLevel::Info, lvl, msg);
}

inline void
log_stats(const VerbosityLevel lvl, const std::string& msg)
{
  log_verbose(VerbosityLevel::Stats, lvl, msg);
}

inline void
log_debug(const VerbosityLevel lvl, const std::string& msg)
{
  log_verbose(VerbosityLevel::Debug, lvl, msg);
}

inline void
log_trace(const VerbosityLevel lvl, const std::string& msg)
{
  log_verbose(VerbosityLevel::Trace, lvl, msg);
}

// =============================================================================
// Unconditional stderr logging
// =============================================================================

inline void
log_warning(const std::string& message)
{
  const std::scoped_lock lock(log_mutex);
  std::cerr << "\x1b[1;33m[WARNING] " << message << "\x1b[0m\n" << std::flush;
}

inline void
log_warning_critical(const std::string& message)
{
  const std::scoped_lock lock(log_mutex);
  std::cerr << "\x1b[1;31m[WARNING] " << message << "\x1b[0m\n" << std::flush;
}

inline void
log_error(const std::string& message)
{
  const std::scoped_lock lock(log_mutex);
  std::cerr << "\x1b[1;31m[ERROR] " << message << "\x1b[0m\n" << std::flush;
}

[[noreturn]] inline void
log_fatal(const std::string& message)
{
  {
    const std::scoped_lock lock(log_mutex);
    std::cerr << "\x1b[1;41m[FATAL] " << message << "\x1b[0m\n";
  }
  std::terminate();
}
}  // namespace starpu_server
