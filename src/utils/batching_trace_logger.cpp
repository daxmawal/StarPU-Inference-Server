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
constexpr unsigned char kAsciiPrintableFloor = 0x20;
constexpr int kRequestEnqueuedSortIndex = -3;
constexpr int kBatchEnqueueSortIndex = -2;
constexpr int kBatchBuildSortIndex = -1;
constexpr int kBatchSubmittedSortIndex = 0;
constexpr std::string_view kProcessName = "StarPU Inference Server";
constexpr std::string_view kRequestEnqueuedTrackName = "request enqueued";
constexpr std::string_view kBatchEnqueueTrackName = "batch";
constexpr std::string_view kBatchBuildTrackName = "dynamic batching";
constexpr std::string_view kBatchSubmittedTrackName = "batch submitted";
constexpr std::string_view kSummarySuffix = "_summary.csv";

enum class FlowDirection : uint8_t {
  None,
  Source,
  Target,
  SourceAndTarget,
};

auto
summary_path_from_trace(const std::filesystem::path& trace_path)
    -> std::filesystem::path
{
  auto summary_path = trace_path;
  auto stem = summary_path.stem().string();
  if (stem.empty()) {
    stem = "batching_trace";
  }
  summary_path.replace_filename(stem + std::string{kSummarySuffix});
  return summary_path;
}

auto
escape_csv_field(std::string_view value) -> std::string
{
  std::string escaped;
  escaped.reserve(value.size());
  for (char ch : value) {
    if (ch == '"') {
      escaped.push_back('"');
    }
    escaped.push_back(ch);
  }
  return escaped;
}

auto
format_request_ids(std::span<const int> request_ids) -> std::string
{
  std::ostringstream oss;
  for (size_t idx = 0; idx < request_ids.size(); ++idx) {
    if (idx > 0) {
      oss << ';';
    }
    oss << request_ids[idx];
  }
  return oss.str();
}

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
  using enum FlowDirection;
  if (direction == None) {
    return;
  }
  const auto bind_id = make_flow_bind_id(batch_id, is_warmup);
  if (!bind_id.has_value()) {
    return;
  }
  const bool emit_flow_out =
      direction == Source || direction == SourceAndTarget;
  const bool emit_flow_in = direction == Target || direction == SourceAndTarget;
  if (!emit_flow_out && !emit_flow_in) {
    return;
  }
  const std::string_view scope = is_warmup ? "warming" : "serving";
  const auto bind_label = std::format("0x{:016X}", *bind_id);
  line << R"(,"id_scope":")" << scope << R"(","id2":{"local":)" << batch_id
       << R"(},"bind_id":")" << bind_label << '"';
  if (emit_flow_out) {
    line << R"(,"flow_out":true)";
  }
  if (emit_flow_in) {
    line << R"(,"flow_in":true)";
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

}  // namespace

namespace detail {

void
RequestTimelineTracker::remember(int request_id, int64_t timestamp_us)
{
  if (request_id < 0) {
    return;
  }
  std::lock_guard lock(mutex_);
  enqueue_times_[request_id] = timestamp_us;
}

auto
RequestTimelineTracker::consume_latest(std::span<const int> request_ids)
    -> std::optional<int64_t>
{
  std::optional<int64_t> latest;
  std::lock_guard lock(mutex_);
  for (int request_id : request_ids) {
    if (request_id < 0) {
      continue;
    }
    const auto timestamp_iter = enqueue_times_.find(request_id);
    if (timestamp_iter == enqueue_times_.end()) {
      continue;
    }
    if (const int64_t timestamp_us = timestamp_iter->second;
        !latest || timestamp_us > *latest) {
      latest = timestamp_us;
    }
    enqueue_times_.erase(timestamp_iter);
  }
  return latest;
}

void
RequestTimelineTracker::reset()
{
  std::lock_guard lock(mutex_);
  enqueue_times_.clear();
}

auto
WorkerLaneManager::assign_lane(int worker_id, Span lane_span) -> Assignment
{
  auto& lanes = worker_lanes_[worker_id];
  if (lanes.empty()) {
    const int thread_id = worker_lane_thread_id(worker_id, /*lane_index=*/0);
    lanes.push_back(LaneState{thread_id, /*last_end_ts=*/0});
  }

  for (size_t idx = 0; idx < lanes.size(); ++idx) {
    auto& lane = lanes[idx];
    if (lane_span.start_ts >= lane.last_end_ts) {
      lane.last_end_ts = lane_span.end_ts;
      const auto lane_index = static_cast<int>(idx);
      return Assignment{
          lane.thread_id, worker_lane_sort_index(worker_id, lane_index),
          lane_index};
    }
  }

  const auto lane_index = static_cast<int>(lanes.size());
  const int thread_id = worker_lane_thread_id(worker_id, lane_index);
  lanes.push_back(LaneState{thread_id, lane_span.end_ts});
  return Assignment{
      thread_id, worker_lane_sort_index(worker_id, lane_index), lane_index};
}

auto
WorkerLaneManager::format_label(
    int worker_id, const Assignment& assignment,
    std::string_view worker_type_str, int device_id) -> std::string
{
  auto label = format_worker_label(worker_id, worker_type_str, device_id);
  if (assignment.lane_index <= 0) {
    return label;
  }
  return std::format("{} #{}", label, assignment.lane_index + 1);
}

void
WorkerLaneManager::reset()
{
  worker_lanes_.clear();
}

auto
WorkerLaneManager::base_thread_id(int worker_id) -> int
{
  return worker_lane_thread_id(worker_id, /*lane_index=*/0);
}

auto
WorkerLaneManager::base_sort_index(int worker_id) -> int
{
  return worker_lane_sort_index(worker_id, /*lane_index=*/0);
}

auto
WorkerLaneManager::worker_lane_sort_index(int worker_id, int lane_index) -> int
{
  const int base_thread_id = worker_lane_thread_id(worker_id, /*lane_index=*/0);
  return base_thread_id * kWorkerLaneSortStride + lane_index;
}

auto
WorkerLaneManager::worker_lane_thread_id(int worker_id, int lane_index) -> int
{
  return kWorkerThreadOffset + worker_id * kWorkerLaneThreadStride + lane_index;
}

bool
TraceFileWriter::open(const std::filesystem::path& file_path)
{
  stream_.open(file_path, std::ios::out | std::ios::trunc);
  reset_state();
  return stream_.is_open();
}

void
TraceFileWriter::close()
{
  if (stream_.is_open()) {
    write_footer();
    stream_.close();
  }
  reset_state();
}

void
TraceFileWriter::write_header()
{
  if (!stream_.is_open()) {
    return;
  }
  stream_ << "{\"traceEvents\":[";
  first_record_ = true;
  header_written_ = true;
  thread_metadata_.clear();
  write_process_metadata();
  ensure_thread_metadata(
      kRequestEnqueuedTrackId, kRequestEnqueuedTrackName,
      kRequestEnqueuedSortIndex);
  ensure_thread_metadata(
      kBatchEnqueueTrackId, kBatchEnqueueTrackName, kBatchEnqueueSortIndex);
  ensure_thread_metadata(
      kBatchBuildTrackId, kBatchBuildTrackName, kBatchBuildSortIndex);
  ensure_thread_metadata(
      kBatchSubmittedTrackId, kBatchSubmittedTrackName,
      kBatchSubmittedSortIndex);
}

void
TraceFileWriter::write_footer()
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
TraceFileWriter::write_line(const std::string& line)
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
TraceFileWriter::write_process_metadata()
{
  const std::string line = std::format(
      R"({{"name":"process_name","ph":"M","ts":0,"pid":{},"args":{{"name":"{}"}}}})",
      kTraceProcessId, kProcessName);
  write_line(line);
}

void
TraceFileWriter::ensure_thread_metadata(
    int thread_id, std::string_view thread_name, int sort_index)
{
  auto& metadata = thread_metadata_[thread_id];
  if (metadata.name != thread_name) {
    metadata.name.assign(thread_name.begin(), thread_name.end());
    const std::string escaped_name = escape_json_string(metadata.name);
    const std::string name_line = std::format(
        R"({{"name":"thread_name","ph":"M","ts":0,"pid":{},"tid":{},"args":{{"name":"{}"}}}})",
        kTraceProcessId, thread_id, escaped_name);
    write_line(name_line);
  }

  if (!metadata.sort_emitted) {
    const std::string sort_line = std::format(
        R"({{"name":"thread_sort_index","ph":"M","ts":0,"pid":{},"tid":{},"args":{{"sort_index":{}}}}})",
        kTraceProcessId, thread_id, sort_index);
    write_line(sort_line);
    metadata.sort_emitted = true;
  }
}

auto
TraceFileWriter::ready() const -> bool
{
  return header_written_ && stream_.is_open();
}

auto
TraceFileWriter::is_open() const -> bool
{
  return stream_.is_open();
}

void
TraceFileWriter::reset_state()
{
  first_record_ = true;
  header_written_ = false;
  thread_metadata_.clear();
}

auto
escape_json_string(std::string_view value) -> std::string
{
  std::string escaped;
  escaped.reserve(value.size());
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        escaped += R"(\")";
        break;
      case '\\':
        escaped += R"(\\)";
        break;
      case '\b':
        escaped += R"(\b)";
        break;
      case '\f':
        escaped += R"(\f)";
        break;
      case '\n':
        escaped += R"(\n)";
        break;
      case '\r':
        escaped += R"(\r)";
        break;
      case '\t':
        escaped += R"(\t)";
        break;
      default:
        if (character < kAsciiPrintableFloor) {
          escaped +=
              std::format("\\u{:04X}", static_cast<unsigned int>(character));
        } else {
          escaped.push_back(static_cast<char>(character));
        }
    }
  }
  return escaped;
}

}  // namespace detail

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

  if (!trace_writer_.open(path)) {
    log_warning(std::format(
        "Failed to open batching trace file '{}'; tracing disabled.",
        file_path_));
    file_path_.clear();
    return;
  }
  configure_summary_writer(path);

  trace_start_us_ = now_us();
  trace_start_initialized_ = true;
  trace_writer_.write_header();

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
  const BatchRecordContext record_context{request_id, kInvalidId, 0};
  const WorkerThreadInfo worker_info{
      kInvalidId, DeviceType::Unknown, kInvalidId};
  write_record(
      BatchingTraceEvent::RequestQueued, model_name, record_context,
      worker_info, std::span<const int>{}, override_timestamp, is_warmup);
}

void
BatchingTraceLogger::log_batch_submitted(const BatchSubmittedLogArgs& args)
{
  const BatchRecordContext record_context{
      kInvalidId, args.batch_id, args.logical_job_count};
  const WorkerThreadInfo worker_info{
      args.worker_id, args.worker_type, args.device_id};
  write_record(
      BatchingTraceEvent::BatchSubmitted, args.model_name, record_context,
      worker_info, args.request_ids, std::nullopt, args.is_warmup);
}

void
BatchingTraceLogger::log_batch_build_span(
    int batch_id, std::string_view model_name, std::size_t batch_size,
    TimeRange schedule, std::span<const int> request_ids, bool is_warmup)
{
  if (!enabled()) {
    return;
  }

  const auto start_ts = relative_timestamp_from_time_point(schedule.start);
  const auto end_ts = relative_timestamp_from_time_point(schedule.end);
  if (!start_ts || !end_ts || *end_ts < *start_ts) {
    return;
  }

  const auto duration = std::max<int64_t>(int64_t{1}, *end_ts - *start_ts);
  write_batch_build_span(
      model_name, batch_id, batch_size,
      BatchSpanTiming{.start_ts = *start_ts, .duration_us = duration},
      request_ids, is_warmup);
}

void
BatchingTraceLogger::log_batch_enqueue_span(
    int batch_id, std::string_view model_name, std::size_t batch_size,
    TimeRange queue_times, std::span<const int> request_ids, bool is_warmup)
{
  if (!enabled()) {
    return;
  }

  const auto start_ts = relative_timestamp_from_time_point(queue_times.start);
  const auto end_ts = relative_timestamp_from_time_point(queue_times.end);
  if (!start_ts.has_value() || !end_ts.has_value()) {
    return;
  }

  int64_t duration = *end_ts - *start_ts;
  if (duration < 0) {
    duration = 0;
  }
  duration = std::max<int64_t>(int64_t{1}, duration);

  write_batch_enqueue_span(
      model_name, batch_id, batch_size,
      BatchSpanTiming{.start_ts = *start_ts, .duration_us = duration},
      request_ids, is_warmup);
}

void
BatchingTraceLogger::log_batch_compute_span(const BatchComputeLogArgs& args)
{
  if (!enabled() || args.worker_id < 0) {
    return;
  }

  const auto start_ts =
      relative_timestamp_from_time_point(args.codelet_times.start);
  const auto end_ts =
      relative_timestamp_from_time_point(args.codelet_times.end);
  if (!start_ts || !end_ts || *end_ts < *start_ts) {
    return;
  }

  const auto duration = std::max<int64_t>(int64_t{1}, *end_ts - *start_ts);
  write_batch_compute_span(BatchComputeWriteArgs{
      .model_name = args.model_name,
      .batch_id = args.batch_id,
      .batch_size = args.batch_size,
      .worker_id = args.worker_id,
      .worker_type = args.worker_type,
      .start_ts = *start_ts,
      .duration_us = duration,
      .is_warmup = args.is_warmup,
      .device_id = args.device_id,
  });
}

void
BatchingTraceLogger::write_record(
    BatchingTraceEvent event, std::string_view model_name,
    const BatchRecordContext& record_context,
    const WorkerThreadInfo& worker_info, std::span<const int> request_ids,
    std::optional<int64_t> override_timestamp, bool is_warmup)
{
  if (!enabled()) {
    return;
  }

  const auto timestamp_us = override_timestamp.has_value()
                                ? *override_timestamp
                                : relative_timestamp_us(now_us());

  if (event == BatchingTraceEvent::RequestQueued) {
    request_timeline_.remember(record_context.request_id, timestamp_us);
  }

  const auto escaped_model = detail::escape_json_string(model_name);
  const auto worker_type_str = device_type_to_string(worker_info.worker_type);
  const bool is_worker_lane = worker_info.worker_id >= 0;
  std::string worker_label;
  int thread_id = kRequestEnqueuedTrackId;
  std::string_view thread_name = kRequestEnqueuedTrackName;
  int sort_index = kRequestEnqueuedSortIndex;
  if (is_worker_lane) {
    worker_label = format_worker_label(
        worker_info.worker_id, worker_type_str, worker_info.device_id);
    thread_name = worker_label;
    sort_index =
        detail::WorkerLaneManager::base_sort_index(worker_info.worker_id);
    thread_id =
        detail::WorkerLaneManager::base_thread_id(worker_info.worker_id);
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
  line << R"({"name":")" << warmup_prefix << event_to_string(event)
       << R"(","cat":"batching","ph":")" << (is_batch_span ? 'X' : 'i')
       << R"(","ts":)" << timestamp_us;
  if (is_batch_span) {
    line << ",\"dur\":1";
  }
  line << ",\"pid\":" << kTraceProcessId << ",\"tid\":" << thread_id;
  const auto span_flow_direction =
      is_batch_span ? FlowDirection::SourceAndTarget : FlowDirection::None;
  append_flow_annotation(
      line, span_flow_direction, record_context.batch_id, is_warmup);
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
      append_numeric("batch_id", record_context.batch_id);
      append_numeric("batch_size", record_context.logical_jobs);
      append_string("model_name", escaped_model);
      break;
    case BatchingTraceEvent::RequestQueued:
      append_numeric("request_id", record_context.request_id);
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
  if (!trace_writer_.ready()) {
    return;
  }
  trace_writer_.ensure_thread_metadata(thread_id, thread_name, sort_index);
  trace_writer_.write_line(line.str());
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
  using enum DeviceType;
  switch (type) {
    case CPU:
      return "cpu";
    case CUDA:
      return "cuda";
    case Unknown:
    default:
      return "unknown";
  }
}

void
BatchingTraceLogger::write_batch_compute_span(const BatchComputeWriteArgs& args)
{
  if (args.worker_id < 0) {
    return;
  }
  int64_t duration_us = args.duration_us;
  if (duration_us <= 0) {
    duration_us = 1;
  }

  const auto escaped_model = detail::escape_json_string(args.model_name);
  const auto worker_type_str = device_type_to_string(args.worker_type);
  const int64_t end_ts = args.start_ts + duration_us;
  const char* warmup_prefix = args.is_warmup ? "warming_" : "";

  std::lock_guard lock(mutex_);
  if (!trace_writer_.ready()) {
    return;
  }

  const auto lane_assignment = worker_lane_manager_.assign_lane(
      args.worker_id, detail::WorkerLaneManager::Span{
                          .start_ts = args.start_ts, .end_ts = end_ts});
  const std::string worker_label = detail::WorkerLaneManager::format_label(
      args.worker_id, lane_assignment, worker_type_str, args.device_id);

  std::ostringstream line;
  line << R"({"name":")" << warmup_prefix << escaped_model
       << R"(","cat":"batching","ph":"X","ts":)" << args.start_ts
       << ",\"dur\":" << duration_us << ",\"pid\":" << kTraceProcessId
       << ",\"tid\":" << lane_assignment.thread_id;
  append_flow_annotation(
      line, FlowDirection::Target, args.batch_id, args.is_warmup);
  line << ",\"args\":{" << "\"" << warmup_prefix
       << "batch_id\":" << args.batch_id << ",\"" << warmup_prefix
       << "batch_size\":" << args.batch_size << ",\"" << warmup_prefix
       << "model_name\":\"" << escaped_model << "\"" << ",\"" << warmup_prefix
       << "worker_id\":" << args.worker_id << ",\"" << warmup_prefix
       << "worker_type\":\"" << worker_type_str << "\"" << ",\""
       << warmup_prefix << "start_ts\":" << args.start_ts << ",\""
       << warmup_prefix << "end_ts\":" << end_ts << "}}";

  trace_writer_.ensure_thread_metadata(
      lane_assignment.thread_id, worker_label, lane_assignment.sort_index);
  trace_writer_.write_line(line.str());
}

void
BatchingTraceLogger::write_batch_enqueue_span(
    std::string_view model_name, int batch_id, std::size_t batch_size,
    BatchSpanTiming timing, std::span<const int> request_ids, bool is_warmup)
{
  if (timing.duration_us <= 0) {
    timing.duration_us = 1;
  }

  auto adjusted_duration = timing.duration_us;
  if (const auto latest_request_ts =
          request_timeline_.consume_latest(request_ids);
      latest_request_ts.has_value()) {
    const int64_t requested_duration = *latest_request_ts - timing.start_ts;
    if (requested_duration > adjusted_duration) {
      adjusted_duration = std::max<int64_t>(int64_t{1}, requested_duration);
    }
  }

  const auto escaped_model = detail::escape_json_string(model_name);
  const char* warmup_prefix = is_warmup ? "warming_" : "";

  std::ostringstream line;
  line << R"({"name":")" << warmup_prefix
       << R"(batch","cat":"batching","ph":"X","ts":)" << timing.start_ts
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
  line << ",\"" << warmup_prefix << "start_ts\":" << timing.start_ts << ",\""
       << warmup_prefix << "end_ts\":" << (timing.start_ts + timing.duration_us)
       << "}}";

  std::lock_guard lock(mutex_);
  if (!trace_writer_.ready()) {
    return;
  }
  trace_writer_.ensure_thread_metadata(
      kBatchEnqueueTrackId, kBatchEnqueueTrackName, kBatchEnqueueSortIndex);
  trace_writer_.write_line(line.str());
}

void
BatchingTraceLogger::write_batch_build_span(
    std::string_view model_name, int batch_id, std::size_t batch_size,
    BatchSpanTiming timing, std::span<const int> request_ids, bool is_warmup)
{
  if (timing.duration_us <= 0) {
    timing.duration_us = 1;
  }

  const auto escaped_model = detail::escape_json_string(model_name);
  const char* warmup_prefix = is_warmup ? "warming_" : "";

  std::ostringstream line;
  line << R"({"name":")" << warmup_prefix
       << R"(batch_build","cat":"batching","ph":"X","ts":)" << timing.start_ts
       << ",\"dur\":" << timing.duration_us << ",\"pid\":" << kTraceProcessId
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
  line << ",\"" << warmup_prefix << "start_ts\":" << timing.start_ts << ",\""
       << warmup_prefix << "end_ts\":" << (timing.start_ts + timing.duration_us)
       << "}}";

  std::lock_guard lock(mutex_);
  if (!trace_writer_.ready()) {
    return;
  }
  trace_writer_.ensure_thread_metadata(
      kBatchBuildTrackId, kBatchBuildTrackName, kBatchBuildSortIndex);
  trace_writer_.write_line(line.str());
}

void
BatchingTraceLogger::log_batch_summary(const BatchSummaryLogArgs& args)
{
  if (!enabled()) {
    return;
  }
  std::lock_guard lock(mutex_);
  if (!summary_stream_.is_open()) {
    return;
  }
  write_summary_line_locked(args);
}

void
BatchingTraceLogger::close_stream_locked()
{
  trace_writer_.close();
  close_summary_writer();
  trace_start_us_ = 0;
  trace_start_initialized_ = false;
  worker_lane_manager_.reset();
  request_timeline_.reset();
}

auto
BatchingTraceLogger::now_us() -> int64_t
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
    std::chrono::high_resolution_clock::time_point time_point) const
    -> std::optional<int64_t>
{
  if (!trace_start_initialized_ ||
      time_point == std::chrono::high_resolution_clock::time_point{}) {
    return std::nullopt;
  }
  const auto absolute_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          time_point.time_since_epoch())
          .count();
  return relative_timestamp_us(absolute_us);
}

void
BatchingTraceLogger::write_summary_line_locked(const BatchSummaryLogArgs& args)
{
  const auto request_ids_string = format_request_ids(args.request_ids);
  const auto escaped_model = escape_csv_field(args.model_name);
  const auto escaped_requests = escape_csv_field(request_ids_string);
  summary_stream_ << args.batch_id << ",\"" << escaped_model << "\","
                  << args.worker_id << ",\""
                  << device_type_to_string(args.worker_type) << "\","
                  << args.device_id << ',' << args.batch_size << ",\""
                  << escaped_requests << "\","
                  << std::format("{:.3f}", args.queue_ms) << ','
                  << std::format("{:.3f}", args.batch_ms) << ','
                  << std::format("{:.3f}", args.submit_ms) << ','
                  << std::format("{:.3f}", args.scheduling_ms) << ','
                  << std::format("{:.3f}", args.codelet_ms) << ','
                  << std::format("{:.3f}", args.inference_ms) << ','
                  << std::format("{:.3f}", args.callback_ms) << ','
                  << std::format("{:.3f}", args.total_ms) << ','
                  << (args.is_warmup ? "true" : "false") << '\n';
}

bool
BatchingTraceLogger::configure_summary_writer(
    const std::filesystem::path& trace_path)
{
  close_summary_writer();
  summary_file_path_ = summary_path_from_trace(trace_path);
  summary_stream_.open(summary_file_path_, std::ios::out | std::ios::trunc);
  if (!summary_stream_.is_open()) {
    log_warning(std::format(
        "Failed to open batching summary file '{}'; summary export disabled.",
        summary_file_path_.string()));
    summary_file_path_.clear();
    return false;
  }
  summary_stream_
      << "batch_id,model_name,worker_id,worker_type,device_id,batch_size,"
         "request_ids,queue_ms,batch_ms,submit_ms,scheduling_ms,codelet_ms,"
         "inference_ms,callback_ms,total_ms,warmup\n";
  return true;
}

void
BatchingTraceLogger::close_summary_writer()
{
  if (summary_stream_.is_open()) {
    summary_stream_.close();
  }
  summary_file_path_.clear();
}

}  // namespace starpu_server
