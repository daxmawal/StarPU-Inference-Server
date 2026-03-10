#pragma once

#include <string>
#include <string_view>

namespace starpu_server { namespace {

const std::string kOverflowLabel{"__overflow__"};
constexpr std::string_view kLabelEscapePrefix{"__label__"};

constexpr int kStatusOk = 0;
constexpr int kStatusCancelled = 1;
constexpr int kStatusUnknown = 2;
constexpr int kStatusInvalidArgument = 3;
constexpr int kStatusDeadlineExceeded = 4;
constexpr int kStatusNotFound = 5;
constexpr int kStatusAlreadyExists = 6;
constexpr int kStatusPermissionDenied = 7;
constexpr int kStatusResourceExhausted = 8;
constexpr int kStatusFailedPrecondition = 9;
constexpr int kStatusAborted = 10;
constexpr int kStatusOutOfRange = 11;
constexpr int kStatusUnimplemented = 12;
constexpr int kStatusInternal = 13;
constexpr int kStatusUnavailable = 14;
constexpr int kStatusDataLoss = 15;
constexpr int kStatusUnauthenticated = 16;

auto
status_code_label(int code) -> std::string
{
  switch (code) {
    case kStatusOk:
      return "OK";
    case kStatusCancelled:
      return "CANCELLED";
    case kStatusUnknown:
      return "UNKNOWN";
    case kStatusInvalidArgument:
      return "INVALID_ARGUMENT";
    case kStatusDeadlineExceeded:
      return "DEADLINE_EXCEEDED";
    case kStatusNotFound:
      return "NOT_FOUND";
    case kStatusAlreadyExists:
      return "ALREADY_EXISTS";
    case kStatusPermissionDenied:
      return "PERMISSION_DENIED";
    case kStatusResourceExhausted:
      return "RESOURCE_EXHAUSTED";
    case kStatusFailedPrecondition:
      return "FAILED_PRECONDITION";
    case kStatusAborted:
      return "ABORTED";
    case kStatusOutOfRange:
      return "OUT_OF_RANGE";
    case kStatusUnimplemented:
      return "UNIMPLEMENTED";
    case kStatusInternal:
      return "INTERNAL";
    case kStatusUnavailable:
      return "UNAVAILABLE";
    case kStatusDataLoss:
      return "DATA_LOSS";
    case kStatusUnauthenticated:
      return "UNAUTHENTICATED";
    default:
      return std::to_string(code);
  }
}

auto
escape_label_value(std::string_view value) -> std::string
{
  if (value == kOverflowLabel || value.starts_with(kLabelEscapePrefix)) {
    std::string escaped;
    escaped.reserve(kLabelEscapePrefix.size() + value.size());
    escaped.append(kLabelEscapePrefix);
    escaped.append(value);
    return escaped;
  }
  return std::string(value);
}

}}  // namespace starpu_server
