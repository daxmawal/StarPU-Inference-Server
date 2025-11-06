#include "batching_trace_logger.hpp"

#include <algorithm>
#include <cstdio>
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
constexpr int kTraceProcessId = 1;
constexpr int kRequestThreadId = 1;
constexpr int kWorkerThreadOffset = 2;
constexpr int kQueueSortIndex = -1;
constexpr std::string_view kProcessName = "StarPU Inference Server";
constexpr std::string_view kRequestThreadName = "task_queue";
}  // namespace

auto
BatchingTraceLogger::instance() -> BatchingTraceLogger&
{
  static BatchingTraceLogger logger;
  return logger;
}

BatchingTraceLogger::~BatchingTraceLogger()
{
  std::lock_guard lock(mutex_);
  close_stream_locked();
}

void
BatchingTraceLogger::configure(bool enabled, std::string file_path)
{
  std::lock_guard lock(mutex_);

  enabled_.store(false, std::memory_order_release);
  close_stream_locked();
  file_path_.clear();

  if (!enabled) {
    return;
  }

  if (file_path.empty()) {
    file_path = "batching_trace.json";
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

  trace_start_us_ = now_us();
  trace_start_initialized_ = true;
  write_header_locked();

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
    std::size_t sample_count, int worker_id, DeviceType worker_type,
    std::chrono::high_resolution_clock::time_point codelet_start,
    std::chrono::high_resolution_clock::time_point codelet_end)
{
  if (!enabled()) {
    return;
  }

  const auto start_ts = relative_timestamp_from_time_point(codelet_start);
  const auto end_ts = relative_timestamp_from_time_point(codelet_end);
  if (start_ts && end_ts && *end_ts >= *start_ts && worker_id >= 0) {
    const auto duration = std::max<int64_t>(int64_t{1}, *end_ts - *start_ts);
    write_batch_compute_span(
        model_name, batch_id, logical_jobs, sample_count, worker_id,
        worker_type, *start_ts, duration);
  }

  write_record(
      BatchingTraceEvent::BatchCompleted, model_name, kInvalidId, batch_id,
      logical_jobs, sample_count, worker_id, worker_type, end_ts, start_ts);
}

void
BatchingTraceLogger::write_record(
    BatchingTraceEvent event, std::string_view model_name, int request_id,
    int batch_id, std::size_t logical_jobs, std::size_t sample_count,
    int worker_id, DeviceType worker_type,
    std::optional<int64_t> override_timestamp,
    std::optional<int64_t> compute_start_ts)
{
  if (!enabled()) {
    return;
  }

  const auto timestamp_us = override_timestamp.has_value()
                                ? *override_timestamp
                                : relative_timestamp_us(now_us());

  const auto escaped_model = escape_json_string(model_name);
  const auto worker_type_str = device_type_to_string(worker_type);
  const bool is_worker_lane = worker_id >= 0;
  const int thread_id =
      is_worker_lane ? worker_id + kWorkerThreadOffset : kRequestThreadId;
  std::string worker_label;
  std::string_view thread_name = kRequestThreadName;
  int sort_index = kQueueSortIndex;
  if (is_worker_lane) {
    worker_label = std::format("worker-{} ({})", worker_id, worker_type_str);
    thread_name = worker_label;
    sort_index = worker_id + kWorkerThreadOffset;
  }

  std::ostringstream line;
  line << "{\"name\":\"" << event_to_string(event)
       << "\",\"cat\":\"batching\",\"ph\":\"i\",\"ts\":" << timestamp_us
       << ",\"pid\":" << kTraceProcessId << ",\"tid\":" << thread_id
       << ",\"args\":{" << "\"request_id\":" << request_id
       << ",\"batch_id\":" << batch_id << ",\"logical_jobs\":" << logical_jobs
       << ",\"sample_count\":" << sample_count << ",\"model_name\":\""
       << escaped_model << "\",\"worker_id\":" << worker_id
       << ",\"worker_type\":\"" << worker_type_str << "\"";
  if (compute_start_ts.has_value()) {
    line << ",\"start_ts\":" << *compute_start_ts;
  }
  line << "}}";

  std::lock_guard lock(mutex_);
  if (!stream_.is_open() || !header_written_) {
    return;
  }
  ensure_thread_metadata_locked(thread_id, thread_name, sort_index);
  write_line_locked(line.str());
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

void
BatchingTraceLogger::write_batch_compute_span(
    std::string_view model_name, int batch_id, std::size_t logical_jobs,
    std::size_t sample_count, int worker_id, DeviceType worker_type,
    int64_t start_ts, int64_t duration_us)
{
  if (worker_id < 0) {
    return;
  }
  if (duration_us <= 0) {
    duration_us = 1;
  }

  const auto escaped_model = escape_json_string(model_name);
  const auto worker_type_str = device_type_to_string(worker_type);
  const int thread_id = worker_id + kWorkerThreadOffset;
  const int sort_index = worker_id + kWorkerThreadOffset;
  const std::string worker_label =
      std::format("worker-{} ({})", worker_id, worker_type_str);

  std::ostringstream line;
  line
      << "{\"name\":\"batch_compute\",\"cat\":\"batching\",\"ph\":\"X\",\"ts\":"
      << start_ts << ",\"dur\":" << duration_us
      << ",\"pid\":" << kTraceProcessId << ",\"tid\":" << thread_id
      << ",\"args\":" << "{\"batch_id\":" << batch_id
      << ",\"logical_jobs\":" << logical_jobs
      << ",\"sample_count\":" << sample_count << ",\"model_name\":\""
      << escaped_model << "\",\"worker_id\":" << worker_id
      << ",\"worker_type\":\"" << worker_type_str
      << "\",\"start_ts\":" << start_ts
      << ",\"end_ts\":" << (start_ts + duration_us) << "}}";

  std::lock_guard lock(mutex_);
  if (!stream_.is_open() || !header_written_) {
    return;
  }
  ensure_thread_metadata_locked(thread_id, worker_label, sort_index);
  write_line_locked(line.str());
}

auto
BatchingTraceLogger::escape_json_string(std::string_view value) -> std::string
{
  std::string escaped;
  escaped.reserve(value.size());
  for (const unsigned char ch : value) {
    switch (ch) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (ch < 0x20) {
          char buffer[7];
          std::snprintf(
              buffer, sizeof(buffer), "\\u%04X", static_cast<unsigned int>(ch));
          escaped.append(buffer, 6);
        } else {
          escaped.push_back(static_cast<char>(ch));
        }
    }
  }
  return escaped;
}

void
BatchingTraceLogger::write_header_locked()
{
  stream_ << "{\"traceEvents\":[";
  first_record_ = true;
  header_written_ = true;
  thread_metadata_.clear();
  write_process_metadata_locked();
  ensure_thread_metadata_locked(
      kRequestThreadId, kRequestThreadName, kQueueSortIndex);
  write_queue_track_marker_locked();
}

void
BatchingTraceLogger::write_footer_locked()
{
  if (!header_written_) {
    return;
  }
  if (!first_record_) {
    stream_ << '\n';
  }
  stream_ << "]\n}\n";
  header_written_ = false;
}

void
BatchingTraceLogger::close_stream_locked()
{
  if (stream_.is_open()) {
    write_footer_locked();
    stream_.close();
  }
  first_record_ = true;
  header_written_ = false;
  thread_metadata_.clear();
  trace_start_us_ = 0;
  trace_start_initialized_ = false;
}

void
BatchingTraceLogger::write_line_locked(const std::string& line)
{
  if (!header_written_) {
    return;
  }
  if (!first_record_) {
    stream_ << ",\n  ";
  } else {
    stream_ << "\n  ";
    first_record_ = false;
  }
  stream_ << line;
}

void
BatchingTraceLogger::write_process_metadata_locked()
{
  const std::string line = std::format(
      "{{\"name\":\"process_name\",\"ph\":\"M\",\"ts\":0,\"pid\":{},\"args\":{{"
      "\"name\":\"{}\"}}}}",
      kTraceProcessId, kProcessName);
  write_line_locked(line);
}

void
BatchingTraceLogger::ensure_thread_metadata_locked(
    int thread_id, std::string_view thread_name, int sort_index)
{
  auto& metadata = thread_metadata_[thread_id];
  if (metadata.name != thread_name) {
    metadata.name.assign(thread_name.begin(), thread_name.end());
    const std::string escaped_name = escape_json_string(metadata.name);
    const std::string name_line = std::format(
        "{{\"name\":\"thread_name\",\"ph\":\"M\",\"ts\":0,\"pid\":{},\"tid\":{}"
        ",\"args\":{{\"name\":\"{}\"}}}}",
        kTraceProcessId, thread_id, escaped_name);
    write_line_locked(name_line);
  }

  if (!metadata.sort_emitted) {
    const std::string sort_line = std::format(
        "{{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"ts\":0,\"pid\":{},"
        "\"tid\":{},\"args\":{{\"sort_index\":{}}}}}",
        kTraceProcessId, thread_id, sort_index);
    write_line_locked(sort_line);
    metadata.sort_emitted = true;
  }
}

void
BatchingTraceLogger::write_queue_track_marker_locked()
{
  const std::string marker = std::format(
      "{{\"name\":\"queue_track_ready\",\"cat\":\"batching\",\"ph\":\"i\","
      "\"ts\":0,"
      "\"pid\":{},\"tid\":{},\"args\":{{\"worker_id\":-1,\"model_name\":\"\"}}}"
      "}",
      kTraceProcessId, kRequestThreadId);
  write_line_locked(marker);
}

auto
BatchingTraceLogger::now_us() const -> int64_t
{
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

auto
BatchingTraceLogger::relative_timestamp_us(int64_t absolute_us) const -> int64_t
{
  if (!trace_start_initialized_) {
    return absolute_us;
  }
  if (absolute_us < trace_start_us_) {
    return 0;
  }
  return absolute_us - trace_start_us_;
}

auto
BatchingTraceLogger::relative_timestamp_from_time_point(
    std::chrono::high_resolution_clock::time_point tp) const
    -> std::optional<int64_t>
{
  if (!trace_start_initialized_ ||
      tp == std::chrono::high_resolution_clock::time_point{}) {
    return std::nullopt;
  }
  const auto absolute_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          tp.time_since_epoch())
          .count();
  return relative_timestamp_us(absolute_us);
}

}  // namespace starpu_server
