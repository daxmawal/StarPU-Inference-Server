#pragma once

#include <concepts>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>

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
// Utility: color and label mapping for verbosity levels
// =============================================================================

inline auto
verbosity_style(const VerbosityLevel level)
    -> std::pair<const char*, const char*>
{
  using enum VerbosityLevel;
  switch (level) {
    case Info:
      return {"\o{33}[1;32m", "[INFO] "};  // Green
    case Stats:
      return {"\o{33}[1;35m", "[STATS] "};  // Magenta
    case Debug:
      return {"\o{33}[1;34m", "[DEBUG] "};  // Blue
    case Trace:
      return {"\o{33}[1;90m", "[TRACE] "};  // Gray
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
    std::cout << color << label << message << "\o{33}[0m\n";
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
  std::cerr << "\o{33}[1;33m[WARNING] " << message << "\o{33}[0m\n";
}

inline void
log_error(const std::string& message)
{
  const std::scoped_lock lock(log_mutex);
  std::cerr << "\o{33}[1;31m[ERROR] " << message << "\o{33}[0m\n";
}

[[noreturn]] inline void
log_fatal(const std::string& message)
{
  {
    const std::scoped_lock lock(log_mutex);
    std::cerr << "\o{33}[1;41m[FATAL] " << message << "\o{33}[0m\n";
  }
  std::terminate();
}
}  // namespace starpu_server
