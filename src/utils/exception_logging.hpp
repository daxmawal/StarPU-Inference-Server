#pragma once

#include <exception>
#include <string>
#include <string_view>
#include <utility>

#include "exceptions.hpp"
#include "logger.hpp"

namespace starpu_server {

struct ExceptionLoggingMessages {
  std::string_view context_prefix;
  std::string_view unknown_message;
};

template <typename Callback>
inline void
run_with_logged_exceptions(
    Callback&& callback,
    const ExceptionLoggingMessages& messages = ExceptionLoggingMessages{})
{
  try {
    std::forward<Callback>(callback)();
  }
  catch (const InferenceEngineException& e) {
    log_error(std::string(messages.context_prefix) + e.what());
  }
  catch (const std::exception& e) {
    log_error(std::string(messages.context_prefix) + e.what());
  }
  catch (...) {
    if (!messages.unknown_message.empty()) {
      log_error(std::string(messages.unknown_message));
    } else {
      log_error(std::string(messages.context_prefix) + "Unknown exception");
    }
  }
}

}  // namespace starpu_server
