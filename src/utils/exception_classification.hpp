#pragma once

#include <cstdint>
#include <exception>
#include <new>
#include <stdexcept>
#include <utility>

#include "exceptions.hpp"

namespace starpu_server {

enum class ExceptionCategory : std::uint8_t {
  InferenceEngine,
  RuntimeError,
  LogicError,
  BadAlloc,
  StdException,
  Unknown,
};

template <
    typename OnInferenceEngineException, typename OnRuntimeError,
    typename OnLogicError, typename OnBadAlloc, typename OnStdException,
    typename OnUnknownException>
inline void
classify_and_handle_exception(
    std::exception_ptr exception,
    OnInferenceEngineException&& on_inference_engine_exception,
    OnRuntimeError&& on_runtime_error, OnLogicError&& on_logic_error,
    OnBadAlloc&& on_bad_alloc, OnStdException&& on_std_exception,
    OnUnknownException&& on_unknown_exception)
{
  if (exception == nullptr) {
    exception = std::current_exception();
  }

  if (exception == nullptr) {
    std::forward<OnUnknownException>(on_unknown_exception)();
    return;
  }

  try {
    std::rethrow_exception(exception);
  }
  catch (const InferenceEngineException& caught_exception) {
    std::forward<OnInferenceEngineException>(on_inference_engine_exception)(
        caught_exception);
  }
  catch (const std::runtime_error& caught_exception) {
    std::forward<OnRuntimeError>(on_runtime_error)(caught_exception);
  }
  catch (const std::logic_error& caught_exception) {
    std::forward<OnLogicError>(on_logic_error)(caught_exception);
  }
  catch (const std::bad_alloc& caught_exception) {
    std::forward<OnBadAlloc>(on_bad_alloc)(caught_exception);
  }
  catch (const std::exception& caught_exception) {
    std::forward<OnStdException>(on_std_exception)(caught_exception);
  }
  catch (...) {
    std::forward<OnUnknownException>(on_unknown_exception)();
  }
}

template <typename OnClassifiedException>
inline void
classify_and_handle_exception(
    std::exception_ptr exception,
    OnClassifiedException&& on_classified_exception)
{
  classify_and_handle_exception(
      exception,
      [&on_classified_exception](
          const InferenceEngineException& caught_exception) {
        std::forward<OnClassifiedException>(on_classified_exception)(
            ExceptionCategory::InferenceEngine, &caught_exception);
      },
      [&on_classified_exception](const std::runtime_error& caught_exception) {
        std::forward<OnClassifiedException>(on_classified_exception)(
            ExceptionCategory::RuntimeError, &caught_exception);
      },
      [&on_classified_exception](const std::logic_error& caught_exception) {
        std::forward<OnClassifiedException>(on_classified_exception)(
            ExceptionCategory::LogicError, &caught_exception);
      },
      [&on_classified_exception](const std::bad_alloc& caught_exception) {
        std::forward<OnClassifiedException>(on_classified_exception)(
            ExceptionCategory::BadAlloc, &caught_exception);
      },
      [&on_classified_exception](const std::exception& caught_exception) {
        std::forward<OnClassifiedException>(on_classified_exception)(
            ExceptionCategory::StdException, &caught_exception);
      },
      [&on_classified_exception]() {
        std::forward<OnClassifiedException>(on_classified_exception)(
            ExceptionCategory::Unknown, nullptr);
      });
}

}  // namespace starpu_server
