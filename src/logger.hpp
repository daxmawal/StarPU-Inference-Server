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

inline void
log_verbose(
    VerbosityLevel level, VerbosityLevel current_level,
    const std::string& message)
{
  if (static_cast<int>(current_level) >= static_cast<int>(level)) {
    switch (level) {
      case VerbosityLevel::Info:
        std::cout << "\033[1;32m[INFO] ";
        break;  // Green
      case VerbosityLevel::Stats:
        std::cout << "\033[1;35m[STATS] ";
        break;  // Magenta
      case VerbosityLevel::Debug:
        std::cout << "\033[1;34m[DEBUG] ";
        break;  // Blue
      case VerbosityLevel::Trace:
        std::cout << "\033[1;90m[TRACE] ";
        break;  // Gray
      default:
        break;
    }
    std::cout << message << "\033[0m" << std::endl;
  }
}

inline void
log_info(VerbosityLevel current_level, const std::string& message)
{
  log_verbose(VerbosityLevel::Info, current_level, message);
}

inline void
log_stats(VerbosityLevel current_level, const std::string& message)
{
  log_verbose(VerbosityLevel::Stats, current_level, message);
}

inline void
log_debug(VerbosityLevel current_level, const std::string& message)
{
  log_verbose(VerbosityLevel::Debug, current_level, message);
}

inline void
log_trace(VerbosityLevel current_level, const std::string& message)
{
  log_verbose(VerbosityLevel::Trace, current_level, message);
}

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