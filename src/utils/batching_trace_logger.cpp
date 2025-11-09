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
constexpr int kRequestEnqueuedTrackId = 1;
constexpr int kBatchEnqueueTrackId = 2;
constexpr int kBatchBuildTrackId = 3;
constexpr int kBatchSubmittedTrackId = 4;
constexpr int kWorkerThreadOffset = 10;
constexpr int kWorkerLaneThreadStride = 1000;
constexpr int kRequestEnqueuedSortIndex = -3;
constexpr int kBatchEnqueueSortIndex = -2;
constexpr int kBatchBuildSortIndex = -1;
constexpr int kBatchSubmittedSortIndex = 0;
constexpr std::string_view kProcessName = "StarPU Inference Server";
constexpr std::string_view kRequestEnqueuedTrackName = "request enqueued";
constexpr std::string_view kBatchEnqueueTrackName = "batch";
constexpr std::string_view kBatchBuildTrackName = "dynamic batching";
constexpr std::string_view kBatchSubmittedTrackName = "batch submitted";

enum class FlowDirection : uint8_t {
  None,
  Source,
  Target,
  SourceAndTarget,
};

auto
make_flow_bind_id(int batch_id, bool is_warmup) -> std::optional<uint64_t>
{
  if (batch_id < 0) {
    return std::nullopt;
  }
  const uint64_t scope_bit = is_warmup ? (uint64_t{1} << 63) : 0;
  return scope_bit | static_cast<uint64_t>(static_cast<uint32_t>(batch_id));
}

void
append_flow_annotation(
    std::ostringstream& line, FlowDirection direction, int batch_id,
    bool is_warmup)
{
  if (direction == FlowDirection::None) {
    return;
  }
  const auto bind_id = make_flow_bind_id(batch_id, is_warmup);
  if (!bind_id) {
    return;
  }
  const bool emit_flow_out = direction == FlowDirection::Source ||
                             direction == FlowDirection::SourceAndTarget;
  const bool emit_flow_in = direction == FlowDirection::Target ||
                            direction == FlowDirection::SourceAndTarget;
  if (!emit_flow_out && !emit_flow_in) {
    return;
  }
  const std::string_view scope = is_warmup ? "warming" : "serving";
  const auto bind_label = std::format("0x{:016X}", *bind_id);
  line << ",\"id_scope\":\"" << scope << "\",\"id2\":{\"local\":" << batch_id
       << "},\"bind_id\":\"" << bind_label << "\"";
  if (emit_flow_out) {
    line << ",\"flow_out\":true";
  }
  if (emit_flow_in) {
    line << ",\"flow_in\":true";
  }
}

auto
format_worker_label(
    int worker_id, std::string_view worker_type_str,
    int device_id) -> std::string
{
  if (device_id >= 0) {
    return std::format(
        "device {} worker {} ({})", device_id, worker_id, worker_type_str);
  }
  return std::format("worker-{} ({})", worker_id, worker_type_str);
}

auto
worker_lane_thread_id(int worker_id, int lane_index) -> int
{
  return kWorkerThreadOffset + worker_id * kWorkerLaneThreadStride + lane_index;
}
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
    int request_id, std::string_view model_name, bool is_warmup,
    std::chrono::high_resolution_clock::time_point event_time)
{
  const auto override_timestamp =
      relative_timestamp_from_time_point(event_time);
  write_record(
      BatchingTraceEvent::RequestQueued, model_name, request_id, kInvalidId, 0,
      kInvalidId, DeviceType::Unknown, std::span<const int>{},
      override_timestamp, is_warmup, kInvalidId);
}

void
BatchingTraceLogger::log_batch_submitted(
    int batch_id, std::string_view model_name, std::size_t logical_jobs,
    int worker_id, DeviceType worker_type, std::span<const int> request_ids,
    bool is_warmup, int device_id)
{
  write_record(
      BatchingTraceEvent::BatchSubmitted, model_name, kInvalidId, batch_id,
      logical_jobs, worker_id, worker_type, request_ids, std::nullopt,
      is_warmup, device_id);
}

void
BatchingTraceLogger::log_batch_build_span(
    int batch_id, std::string_view model_name, std::size_t batch_size,
    std::chrono::high_resolution_clock::time_point start_time,
    std::chrono::high_resolution_clock::time_point end_time,
    std::span<const int> request_ids, bool is_warmup)
{
  if (!enabled()) {
    return;
  }

  const auto start_ts = relative_timestamp_from_time_point(start_time);
  const auto end_ts = relative_timestamp_from_time_point(end_time);
  if (!start_ts || !end_ts || *end_ts < *start_ts) {
    return;
  }

  const auto duration = std::max<int64_t>(int64_t{1}, *end_ts - *start_ts);
  write_batch_build_span(
      model_name, batch_id, batch_size, *start_ts, duration, request_ids,
      is_warmup);
}

void
BatchingTraceLogger::log_batch_enqueue_span(
    int batch_id, std::string_view model_name, std::size_t batch_size,
    std::chrono::high_resolution_clock::time_point start_time,
    std::chrono::high_resolution_clock::time_point end_time,
    std::span<const int> request_ids, bool is_warmup)
{
  if (!enabled()) {
    return;
  }

  const auto start_ts = relative_timestamp_from_time_point(start_time);
  const auto end_ts = relative_timestamp_from_time_point(end_time);
  if (!start_ts || !end_ts) {
    return;
  }

  int64_t duration = *end_ts - *start_ts;
  if (duration < 0) {
    duration = 0;
  }
  duration = std::max<int64_t>(int64_t{1}, duration);

  write_batch_enqueue_span(
      model_name, batch_id, batch_size, *start_ts, duration, request_ids,
      is_warmup);
}

void
BatchingTraceLogger::log_batch_compute_span(
    int batch_id, std::string_view model_name, std::size_t batch_size,
    int worker_id, DeviceType worker_type,
    std::chrono::high_resolution_clock::time_point codelet_start,
    std::chrono::high_resolution_clock::time_point codelet_end, bool is_warmup,
    int device_id)
{
  if (!enabled() || worker_id < 0) {
    return;
  }

  const auto start_ts = relative_timestamp_from_time_point(codelet_start);
  const auto end_ts = relative_timestamp_from_time_point(codelet_end);
  if (!start_ts || !end_ts || *end_ts < *start_ts) {
    return;
  }

  const auto duration = std::max<int64_t>(int64_t{1}, *end_ts - *start_ts);
  write_batch_compute_span(
      model_name, batch_id, batch_size, worker_id, worker_type, *start_ts,
      duration, is_warmup, device_id);
}

void
BatchingTraceLogger::write_record(
    BatchingTraceEvent event, std::string_view model_name, int request_id,
    int batch_id, std::size_t logical_jobs, int worker_id,
    DeviceType worker_type, std::span<const int> request_ids,
    std::optional<int64_t> override_timestamp, bool is_warmup, int device_id)
{
  if (!enabled()) {
    return;
  }

  const auto timestamp_us = override_timestamp.has_value()
                                ? *override_timestamp
                                : relative_timestamp_us(now_us());

  if (event == BatchingTraceEvent::RequestQueued) {
    remember_request_enqueue_timestamp(request_id, timestamp_us);
  }

  const auto escaped_model = escape_json_string(model_name);
  const auto worker_type_str = device_type_to_string(worker_type);
  const bool is_worker_lane = worker_id >= 0;
  std::string worker_label;
  int thread_id = kRequestEnqueuedTrackId;
  std::string_view thread_name = kRequestEnqueuedTrackName;
  int sort_index = kRequestEnqueuedSortIndex;
  if (is_worker_lane) {
    worker_label = format_worker_label(worker_id, worker_type_str, device_id);
    thread_name = worker_label;
    constexpr int kBaseLaneIndex = 0;
    sort_index = worker_lane_sort_index(worker_id, kBaseLaneIndex);
    thread_id = worker_lane_thread_id(worker_id, kBaseLaneIndex);
  } else {
    switch (event) {
      case BatchingTraceEvent::RequestQueued:
        thread_id = kRequestEnqueuedTrackId;
        thread_name = kRequestEnqueuedTrackName;
        sort_index = kRequestEnqueuedSortIndex;
        break;
      case BatchingTraceEvent::BatchSubmitted:
        thread_id = kBatchSubmittedTrackId;
        thread_name = kBatchSubmittedTrackName;
        sort_index = kBatchSubmittedSortIndex;
        break;
    }
  }

  const char* warmup_prefix = is_warmup ? "warming_" : "";

  const bool is_batch_span = event == BatchingTraceEvent::BatchSubmitted;

  std::ostringstream line;
  line << "{\"name\":\"" << warmup_prefix << event_to_string(event)
       << "\",\"cat\":\"batching\",\"ph\":\"" << (is_batch_span ? 'X' : 'i')
       << "\",\"ts\":" << timestamp_us;
  if (is_batch_span) {
    line << ",\"dur\":1";
  }
  line << ",\"pid\":" << kTraceProcessId << ",\"tid\":" << thread_id;
  const auto span_flow_direction =
      is_batch_span ? FlowDirection::SourceAndTarget : FlowDirection::None;
  append_flow_annotation(line, span_flow_direction, batch_id, is_warmup);
  line << ",\"args\":{";

  bool first_arg = true;
  const auto append_delimiter = [&]() {
    if (!first_arg) {
      line << ',';
    }
    first_arg = false;
  };
  const auto append_numeric = [&](std::string_view key, auto value) {
    append_delimiter();
    line << "\"" << warmup_prefix << key << "\":" << value;
  };
  const auto append_string = [&](std::string_view key, std::string_view value) {
    append_delimiter();
    line << "\"" << warmup_prefix << key << "\":\"" << value << "\"";
  };

  switch (event) {
    case BatchingTraceEvent::BatchSubmitted:
      append_numeric("batch_id", batch_id);
      append_numeric("batch_size", logical_jobs);
      append_string("model_name", escaped_model);
      break;
    case BatchingTraceEvent::RequestQueued:
      append_numeric("request_id", request_id);
      append_string("model_name", escaped_model);
      break;
  }

  if (!request_ids.empty()) {
    append_delimiter();
    line << "\"" << warmup_prefix << "request_ids\":[";
    for (size_t idx = 0; idx < request_ids.size(); ++idx) {
      if (idx > 0) {
        line << ',';
      }
      line << request_ids[idx];
    }
    line << "]";
  }

  line << "}}";

  std::lock_guard lock(mutex_);
  if (!stream_.is_open() || !header_written_) {
    return;
  }
  ensure_thread_metadata_locked(thread_id, thread_name, sort_index);
  write_line_locked(line.str());
}

void
BatchingTraceLogger::remember_request_enqueue_timestamp(
    int request_id, int64_t timestamp_us)
{
  if (request_id < 0) {
    return;
  }
  std::lock_guard lock(request_time_mutex_);
  request_enqueue_times_[request_id] = timestamp_us;
}

auto
BatchingTraceLogger::consume_latest_request_enqueue_timestamp(
    std::span<const int> request_ids) -> std::optional<int64_t>
{
  std::optional<int64_t> latest;
  std::lock_guard lock(request_time_mutex_);
  for (int request_id : request_ids) {
    if (request_id < 0) {
      continue;
    }
    const auto it = request_enqueue_times_.find(request_id);
    if (it == request_enqueue_times_.end()) {
      continue;
    }
    const int64_t ts = it->second;
    if (!latest || ts > *latest) {
      latest = ts;
    }
    request_enqueue_times_.erase(it);
  }
  return latest;
}

auto
BatchingTraceLogger::event_to_string(BatchingTraceEvent event)
    -> std::string_view
{
  switch (event) {
    case BatchingTraceEvent::RequestQueued:
      return "request_enqueued";
    case BatchingTraceEvent::BatchSubmitted:
      return "batch_submitted";
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
    std::string_view model_name, int batch_id, std::size_t batch_size,
    int worker_id, DeviceType worker_type, int64_t start_ts,
    int64_t duration_us, bool is_warmup, int device_id)
{
  if (worker_id < 0) {
    return;
  }
  if (duration_us <= 0) {
    duration_us = 1;
  }

  const auto escaped_model = escape_json_string(model_name);
  const auto worker_type_str = device_type_to_string(worker_type);
  const int64_t end_ts = start_ts + duration_us;
  const char* warmup_prefix = is_warmup ? "warming_" : "";

  std::lock_guard lock(mutex_);
  if (!stream_.is_open() || !header_written_) {
    return;
  }

  const auto lane_assignment =
      assign_worker_lane_locked(worker_id, start_ts, end_ts);
  const std::string worker_label = format_worker_lane_label(
      worker_id, worker_type_str, device_id, lane_assignment.lane_index);

  std::ostringstream line;
  line << "{\"name\":\"" << warmup_prefix
       << "inference\",\"cat\":\"batching\",\"ph\":\"X\",\"ts\":" << start_ts
       << ",\"dur\":" << duration_us << ",\"pid\":" << kTraceProcessId
       << ",\"tid\":" << lane_assignment.thread_id;
  append_flow_annotation(line, FlowDirection::Target, batch_id, is_warmup);
  line << ",\"args\":{" << "\"" << warmup_prefix << "batch_id\":" << batch_id
       << ",\"" << warmup_prefix << "batch_size\":" << batch_size << ",\""
       << warmup_prefix << "model_name\":\"" << escaped_model << "\"" << ",\""
       << warmup_prefix << "worker_id\":" << worker_id << ",\"" << warmup_prefix
       << "worker_type\":\"" << worker_type_str << "\"" << ",\""
       << warmup_prefix << "start_ts\":" << start_ts << ",\"" << warmup_prefix
       << "end_ts\":" << (start_ts + duration_us) << "}}";

  ensure_thread_metadata_locked(
      lane_assignment.thread_id, worker_label, lane_assignment.sort_index);
  write_line_locked(line.str());
}

auto
BatchingTraceLogger::assign_worker_lane_locked(
    int worker_id, int64_t start_ts, int64_t end_ts) -> WorkerLaneAssignment
{
  auto& lanes = worker_lanes_[worker_id];
  if (lanes.empty()) {
    lanes.push_back(WorkerLaneState{
        worker_lane_thread_id(worker_id, /*lane_index=*/0),
        /*last_end_ts=*/0});
  }

  int lane_index = 0;
  for (auto& lane : lanes) {
    if (start_ts >= lane.last_end_ts) {
      lane.last_end_ts = end_ts;
      return WorkerLaneAssignment{
          lane.thread_id, worker_lane_sort_index(worker_id, lane_index),
          lane_index};
    }
    ++lane_index;
  }

  lane_index = static_cast<int>(lanes.size());
  const int thread_id = worker_lane_thread_id(worker_id, lane_index);
  lanes.push_back(WorkerLaneState{thread_id, end_ts});
  return WorkerLaneAssignment{
      thread_id, worker_lane_sort_index(worker_id, lane_index), lane_index};
}

auto
BatchingTraceLogger::worker_lane_sort_index(int worker_id, int lane_index)
    -> int
{
  const int base = worker_lane_thread_id(worker_id, /*lane_index=*/0);
  return base * kWorkerLaneSortStride + lane_index;
}

auto
BatchingTraceLogger::format_worker_lane_label(
    int worker_id, std::string_view worker_type_str, int device_id,
    int lane_index) -> std::string
{
  auto label = format_worker_label(worker_id, worker_type_str, device_id);
  if (lane_index <= 0) {
    return label;
  }
  return std::format("{} #{}", label, lane_index + 1);
}

void
BatchingTraceLogger::write_batch_enqueue_span(
    std::string_view model_name, int batch_id, std::size_t batch_size,
    int64_t start_ts, int64_t duration_us, std::span<const int> request_ids,
    bool is_warmup)
{
  if (duration_us <= 0) {
    duration_us = 1;
  }

  auto adjusted_duration = duration_us;
  const auto latest_request_ts =
      consume_latest_request_enqueue_timestamp(request_ids);
  if (latest_request_ts) {
    const int64_t requested_duration = *latest_request_ts - start_ts;
    if (requested_duration > adjusted_duration) {
      adjusted_duration = std::max<int64_t>(int64_t{1}, requested_duration);
    }
  }

  const auto escaped_model = escape_json_string(model_name);
  const char* warmup_prefix = is_warmup ? "warming_" : "";

  std::ostringstream line;
  line << "{\"name\":\"" << warmup_prefix
       << "batch\",\"cat\":\"batching\",\"ph\":\"X\",\"ts\":" << start_ts
       << ",\"dur\":" << adjusted_duration << ",\"pid\":" << kTraceProcessId
       << ",\"tid\":" << kBatchEnqueueTrackId;
  line << ",\"args\":{" << "\"" << warmup_prefix << "batch_id\":" << batch_id
       << ",\"" << warmup_prefix << "batch_size\":" << batch_size << ",\""
       << warmup_prefix << "model_name\":\"" << escaped_model << "\"";
  if (!request_ids.empty()) {
    line << ",\"" << warmup_prefix << "request_ids\":[";
    for (size_t idx = 0; idx < request_ids.size(); ++idx) {
      if (idx > 0) {
        line << ',';
      }
      line << request_ids[idx];
    }
    line << "]";
  }
  line << ",\"" << warmup_prefix << "start_ts\":" << start_ts << ",\""
       << warmup_prefix << "end_ts\":" << (start_ts + duration_us) << "}}";

  std::lock_guard lock(mutex_);
  if (!stream_.is_open() || !header_written_) {
    return;
  }
  ensure_thread_metadata_locked(
      kBatchEnqueueTrackId, kBatchEnqueueTrackName, kBatchEnqueueSortIndex);
  write_line_locked(line.str());
}

void
BatchingTraceLogger::write_batch_build_span(
    std::string_view model_name, int batch_id, std::size_t batch_size,
    int64_t start_ts, int64_t duration_us, std::span<const int> request_ids,
    bool is_warmup)
{
  if (duration_us <= 0) {
    duration_us = 1;
  }

  const auto escaped_model = escape_json_string(model_name);
  const char* warmup_prefix = is_warmup ? "warming_" : "";

  std::ostringstream line;
  line << "{\"name\":\"" << warmup_prefix
       << "batch_build\",\"cat\":\"batching\",\"ph\":\"X\",\"ts\":" << start_ts
       << ",\"dur\":" << duration_us << ",\"pid\":" << kTraceProcessId
       << ",\"tid\":" << kBatchBuildTrackId;
  append_flow_annotation(line, FlowDirection::Source, batch_id, is_warmup);
  line << ",\"args\":{" << "\"" << warmup_prefix << "batch_id\":" << batch_id
       << ",\"" << warmup_prefix << "batch_size\":" << batch_size << ",\""
       << warmup_prefix << "model_name\":\"" << escaped_model << "\"";
  if (!request_ids.empty()) {
    line << ",\"" << warmup_prefix << "request_ids\":[";
    for (size_t idx = 0; idx < request_ids.size(); ++idx) {
      if (idx > 0) {
        line << ',';
      }
      line << request_ids[idx];
    }
    line << "]";
  }
  line << ",\"" << warmup_prefix << "start_ts\":" << start_ts << ",\""
       << warmup_prefix << "end_ts\":" << (start_ts + duration_us) << "}}";

  std::lock_guard lock(mutex_);
  if (!stream_.is_open() || !header_written_) {
    return;
  }
  ensure_thread_metadata_locked(
      kBatchBuildTrackId, kBatchBuildTrackName, kBatchBuildSortIndex);
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
      kRequestEnqueuedTrackId, kRequestEnqueuedTrackName,
      kRequestEnqueuedSortIndex);
  ensure_thread_metadata_locked(
      kBatchEnqueueTrackId, kBatchEnqueueTrackName, kBatchEnqueueSortIndex);
  ensure_thread_metadata_locked(
      kBatchBuildTrackId, kBatchBuildTrackName, kBatchBuildSortIndex);
  ensure_thread_metadata_locked(
      kBatchSubmittedTrackId, kBatchSubmittedTrackName,
      kBatchSubmittedSortIndex);
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
  worker_lanes_.clear();
  {
    std::lock_guard request_lock(request_time_mutex_);
    request_enqueue_times_.clear();
  }
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
