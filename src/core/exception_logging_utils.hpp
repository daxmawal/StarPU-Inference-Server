#pragma once

#include <exception>
#include <string_view>

#include "utils/logger.hpp"

namespace starpu_server {

struct ExceptionLoggingMessages {
  std::string_view context_prefix;
  std::string_view unknown_message;
};

class InferenceEngineException;

template <typename Callback>
void
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
  catch (const std::bad_alloc& e) {
    log_error(std::string(messages.context_prefix) + e.what());
  }
  catch (const std::runtime_error& e) {
    log_error(std::string(messages.context_prefix) + e.what());
  }
  catch (const std::logic_error& e) {
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
