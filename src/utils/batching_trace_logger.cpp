#include "batching_trace_logger.hpp"

#include <filesystem>
#include <format>
#include <sstream>
#include <utility>

#include "device_type.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"

namespace starpu_server {

namespace {
constexpr int kInvalidId = -1;

auto
format_model_name(std::string_view name) -> std::string
{
  if (name.empty()) {
    return "\"\"";
  }
  std::string formatted;
  formatted.reserve(name.size() + 2);
  formatted.push_back('"');
  for (const char ch : name) {
    if (ch == '"') {
      formatted.push_back('\\');
    }
    formatted.push_back(ch);
  }
  formatted.push_back('"');
  return formatted;
}
}  // namespace

auto
BatchingTraceLogger::instance() -> BatchingTraceLogger&
{
  static BatchingTraceLogger logger;
  return logger;
}

void
BatchingTraceLogger::configure(bool enabled, std::string file_path)
{
  std::lock_guard lock(mutex_);

  enabled_.store(false, std::memory_order_release);
  if (stream_.is_open()) {
    stream_.close();
  }
  file_path_.clear();

  if (!enabled) {
    return;
  }

  if (file_path.empty()) {
    file_path = "batching_trace.log";
  }

  file_path_ = std::move(file_path);

  const std::filesystem::path path{file_path_};
  if (!path.parent_path().empty()) {
    std::error_code dir_ec;
    std::filesystem::create_directories(path.parent_path(), dir_ec);
    if (dir_ec) {
      log_warning(std::format(
          "Failed to create directory '{}' for batching trace ({}).",
          path.parent_path().string(), dir_ec.message()));
    }
  }

  stream_.open(path, std::ios::out | std::ios::trunc);
  if (!stream_.is_open()) {
    log_warning(std::format(
        "Failed to open batching trace file '{}'; tracing disabled.",
        file_path_));
    file_path_.clear();
    return;
  }

  stream_ << "timestamp_us,event,request_id,batch_id,logical_jobs,"
          << "sample_count,model_name,worker_id,worker_type\n";

  enabled_.store(true, std::memory_order_release);
}

void
BatchingTraceLogger::configure_from_runtime(const RuntimeConfig& cfg)
{
  configure(cfg.batching.trace_enabled, cfg.batching.trace_file_path);
}

auto
BatchingTraceLogger::enabled() const -> bool
{
  return enabled_.load(std::memory_order_acquire);
}

void
BatchingTraceLogger::log_request_enqueued(
    int request_id, std::string_view model_name)
{
  write_record(
      BatchingTraceEvent::RequestQueued, model_name, request_id, kInvalidId, 0,
      0, kInvalidId, DeviceType::Unknown);
}

void
BatchingTraceLogger::log_request_assigned_to_batch(
    int request_id, int batch_id, std::string_view model_name,
    std::size_t logical_jobs, std::size_t sample_count, int worker_id,
    DeviceType worker_type)
{
  write_record(
      BatchingTraceEvent::RequestAssigned, model_name, request_id, batch_id,
      logical_jobs, sample_count, worker_id, worker_type);
}

void
BatchingTraceLogger::log_batch_submitted(
    int batch_id, std::string_view model_name, std::size_t logical_jobs,
    std::size_t sample_count, int worker_id, DeviceType worker_type)
{
  write_record(
      BatchingTraceEvent::BatchSubmitted, model_name, kInvalidId, batch_id,
      logical_jobs, sample_count, worker_id, worker_type);
}

void
BatchingTraceLogger::log_batch_completed(
    int batch_id, std::string_view model_name, std::size_t logical_jobs,
    std::size_t sample_count, int worker_id, DeviceType worker_type)
{
  write_record(
      BatchingTraceEvent::BatchCompleted, model_name, kInvalidId, batch_id,
      logical_jobs, sample_count, worker_id, worker_type);
}

void
BatchingTraceLogger::write_record(
    BatchingTraceEvent event, std::string_view model_name, int request_id,
    int batch_id, std::size_t logical_jobs, std::size_t sample_count,
    int worker_id, DeviceType worker_type)
{
  if (!enabled()) {
    return;
  }

  const auto timestamp_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();

  std::ostringstream line;
  line << timestamp_us << ',' << event_to_string(event) << ',' << request_id
       << ',' << batch_id << ',' << logical_jobs << ',' << sample_count << ','
       << format_model_name(model_name) << ',' << worker_id << ','
       << device_type_to_string(worker_type) << '\n';

  std::lock_guard lock(mutex_);
  if (!stream_.is_open()) {
    return;
  }
  stream_ << line.str();
}

auto
BatchingTraceLogger::event_to_string(BatchingTraceEvent event)
    -> std::string_view
{
  switch (event) {
    case BatchingTraceEvent::RequestQueued:
      return "request_enqueued";
    case BatchingTraceEvent::RequestAssigned:
      return "request_assigned";
    case BatchingTraceEvent::BatchSubmitted:
      return "batch_submitted";
    case BatchingTraceEvent::BatchCompleted:
      return "batch_completed";
  }
  return "unknown";
}

auto
BatchingTraceLogger::device_type_to_string(DeviceType type) -> std::string_view
{
  switch (type) {
    case DeviceType::CPU:
      return "cpu";
    case DeviceType::CUDA:
      return "cuda";
    case DeviceType::Unknown:
    default:
      return "unknown";
  }
}

}  // namespace starpu_server
