#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

enum class VerbosityLevel {
  Silent = 0,
  Info = 1,
  Stats = 2,
  Debug = 3,
  Trace = 4
};

// =============================================================================
// Utility: color and label mapping for verbosity levels
// =============================================================================
inline std::pair<const char*, const char*>
verbosity_style(VerbosityLevel level)
{
  switch (level) {
    case VerbosityLevel::Info:
      return {"\033[1;32m", "[INFO] "};  // Green
    case VerbosityLevel::Stats:
      return {"\033[1;35m", "[STATS] "};  // Magenta
    case VerbosityLevel::Debug:
      return {"\033[1;34m", "[DEBUG] "};  // Blue
    case VerbosityLevel::Trace:
      return {"\033[1;90m", "[TRACE] "};  // Gray
    default:
      return {"", ""};
  }
}

// =============================================================================
// Verbosity-controlled logging
// =============================================================================
inline void
log_verbose(
    VerbosityLevel level, VerbosityLevel current_level,
    const std::string& message)
{
  if (static_cast<int>(current_level) >= static_cast<int>(level)) {
    auto [color, label] = verbosity_style(level);
    std::cout << color << label << message << "\033[0m" << std::endl;
  }
}

// =============================================================================
// Shortcut wrappers
// =============================================================================
inline void
log_info(VerbosityLevel lvl, const std::string& msg)
{
  log_verbose(VerbosityLevel::Info, lvl, msg);
}
inline void
log_stats(VerbosityLevel lvl, const std::string& msg)
{
  log_verbose(VerbosityLevel::Stats, lvl, msg);
}
inline void
log_debug(VerbosityLevel lvl, const std::string& msg)
{
  log_verbose(VerbosityLevel::Debug, lvl, msg);
}
inline void
log_trace(VerbosityLevel lvl, const std::string& msg)
{
  log_verbose(VerbosityLevel::Trace, lvl, msg);
}

// =============================================================================
// Unconditional stderr logging
// =============================================================================
inline void
log_warning(const std::string& message)
{
  std::cerr << "\033[1;33m[WARNING] " << message << "\033[0m" << std::endl;
}

inline void
log_error(const std::string& message)
{
  std::cerr << "\033[1;31m[ERROR] " << message << "\033[0m" << std::endl;
}

inline void
log_fatal(const std::string& message)
{
  std::cerr << "\033[1;41m[FATAL] " << message << "\033[0m" << std::endl;
  std::exit(EXIT_FAILURE);
}
