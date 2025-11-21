#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "device_type.hpp"

namespace starpu_server {

struct RuntimeConfig;

enum class BatchingTraceEvent : uint8_t { RequestQueued, BatchSubmitted };

namespace detail {

class RequestTimelineTracker {
 public:
  void remember(int request_id, int64_t timestamp_us);
  [[nodiscard]] auto consume_latest(std::span<const int> request_ids)
      -> std::optional<int64_t>;
  void reset();

 private:
  std::mutex mutex_;
  std::unordered_map<int, int64_t> enqueue_times_;
};

class WorkerLaneManager {
 public:
  struct Span {
    int64_t start_ts;
    int64_t end_ts;
  };
  struct Assignment {
    int thread_id;
    int sort_index;
    int lane_index;
  };

  auto assign_lane(int worker_id, Span lane_span) -> Assignment;
  [[nodiscard]] static auto format_label(
      int worker_id, const Assignment& assignment,
      std::string_view worker_type_str, int device_id) -> std::string;
  [[nodiscard]] static auto base_thread_id(int worker_id) -> int;
  [[nodiscard]] static auto base_sort_index(int worker_id) -> int;
  void reset();

 private:
  struct LaneState {
    int thread_id;
    int64_t last_end_ts;
  };
  struct LaneIndex {
    int value;
  };

  static constexpr int kWorkerLaneSortStride = 1000;
  static auto worker_lane_sort_index(int worker_id, LaneIndex lane_index)
      -> int;
  static auto worker_lane_thread_id(int worker_id, LaneIndex lane_index) -> int;

  std::unordered_map<int, std::vector<LaneState>> worker_lanes_;
};

class TraceFileWriter {
 public:
  auto open(const std::filesystem::path& file_path) -> bool;
  void close();
  void write_header();
  void write_footer();
  void write_line(const std::string& line);
  void write_process_metadata();
  void ensure_thread_metadata(
      int thread_id, std::string_view thread_name, int sort_index);
  [[nodiscard]] auto ready() const -> bool;
  [[nodiscard]] auto is_open() const -> bool;
  void reset_state();

 private:
  struct ThreadMetadata {
    std::string name;
    bool sort_emitted = false;
  };

  std::ofstream stream_;
  bool first_record_{true};
  bool header_written_{false};
  std::unordered_map<int, ThreadMetadata> thread_metadata_;
};

auto escape_json_string(std::string_view value) -> std::string;

}  // namespace detail

class BatchingTraceLogger {
 public:
  struct TimeRange {
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
  };
  struct BatchSpanTiming {
    int64_t start_ts;
    int64_t duration_us;
  };
  struct BatchRecordContext {
    int request_id;
    int batch_id;
    std::size_t logical_jobs;
  };
  struct WorkerThreadInfo {
    int worker_id;
    DeviceType worker_type;
    int device_id;
  };
  struct BatchSubmittedLogArgs {
    int batch_id;
    std::string_view model_name;
    std::size_t logical_job_count = 0;
    DeviceType worker_type = DeviceType::Unknown;
    int worker_id = -1;
    std::span<const int> request_ids;
    bool is_warmup = false;
    int device_id = -1;
  };
  struct BatchComputeLogArgs {
    int batch_id;
    std::string_view model_name;
    std::size_t batch_size;
    int worker_id;
    DeviceType worker_type = DeviceType::Unknown;
    TimeRange codelet_times{};
    bool is_warmup = false;
    int device_id = -1;
  };
  struct BatchComputeWriteArgs {
    std::string_view model_name;
    int batch_id;
    std::size_t batch_size;
    int worker_id;
    DeviceType worker_type;
    int64_t start_ts;
    int64_t duration_us;
    bool is_warmup;
    int device_id;
  };
  struct BatchSummaryLogArgs {
    int batch_id;
    std::string_view model_name;
    std::size_t batch_size;
    std::span<const int> request_ids;
    std::span<const int64_t> request_arrival_us;
    int worker_id;
    DeviceType worker_type;
    int device_id;
    double queue_ms;
    double batch_ms;
    double submit_ms;
    double scheduling_ms;
    double codelet_ms;
    double inference_ms;
    double callback_ms;
    double total_ms;
    bool is_warmup = false;
  };

  static auto instance() -> BatchingTraceLogger&;
  BatchingTraceLogger() = default;
  ~BatchingTraceLogger();
  BatchingTraceLogger(const BatchingTraceLogger&) = delete;
  auto operator=(const BatchingTraceLogger&) -> BatchingTraceLogger& = delete;
  BatchingTraceLogger(BatchingTraceLogger&&) = delete;
  auto operator=(BatchingTraceLogger&&) -> BatchingTraceLogger& = delete;

  void configure(bool enabled, std::string file_path);
  void configure_from_runtime(const RuntimeConfig& cfg);

  [[nodiscard]] auto enabled() const -> bool;

  void log_request_enqueued(
      int request_id, std::string_view model_name, bool is_warmup = false,
      std::chrono::high_resolution_clock::time_point event_time = {});
  void log_batch_submitted(const BatchSubmittedLogArgs& args);
  void log_batch_build_span(
      int batch_id, std::string_view model_name, std::size_t batch_size,
      TimeRange schedule, std::span<const int> request_ids = {},
      bool is_warmup = false);
  void log_batch_enqueue_span(
      int batch_id, std::string_view model_name, std::size_t batch_size,
      TimeRange queue_times, std::span<const int> request_ids = {},
      bool is_warmup = false);
  void log_batch_compute_span(const BatchComputeLogArgs& args);
  void log_batch_summary(const BatchSummaryLogArgs& args);
  [[nodiscard]] auto summary_file_path() const
      -> std::optional<std::filesystem::path>;

 private:
  void write_record(
      BatchingTraceEvent event, std::string_view model_name,
      const BatchRecordContext& record_context,
      const WorkerThreadInfo& worker_info, std::span<const int> request_ids,
      std::optional<int64_t> override_timestamp = std::nullopt,
      bool is_warmup = false);
  void write_batch_compute_span(const BatchComputeWriteArgs& args);
  void write_batch_enqueue_span(
      std::string_view model_name, int batch_id, std::size_t batch_size,
      BatchSpanTiming timing, std::span<const int> request_ids, bool is_warmup);
  void write_batch_build_span(
      std::string_view model_name, int batch_id, std::size_t batch_size,
      BatchSpanTiming timing, std::span<const int> request_ids, bool is_warmup);
  void write_summary_line_locked(const BatchSummaryLogArgs& args);
  auto configure_summary_writer(const std::filesystem::path& trace_path)
      -> bool;
  void close_summary_writer();
  [[nodiscard]] static auto event_to_string(BatchingTraceEvent event)
      -> std::string_view;
  [[nodiscard]] static auto device_type_to_string(DeviceType type)
      -> std::string_view;
  [[nodiscard]] auto relative_timestamp_from_time_point(
      std::chrono::high_resolution_clock::time_point time_point) const
      -> std::optional<int64_t>;

  void close_stream_locked();
  [[nodiscard]] static auto now_us() -> int64_t;
  [[nodiscard]] auto relative_timestamp_us(int64_t absolute_us) const
      -> int64_t;

  mutable std::mutex mutex_;
  std::atomic<bool> enabled_{false};
  std::string file_path_;
  int64_t trace_start_us_{0};
  bool trace_start_initialized_{false};

  detail::TraceFileWriter trace_writer_;
  detail::RequestTimelineTracker request_timeline_;
  detail::WorkerLaneManager worker_lane_manager_;
  std::ofstream summary_stream_;
  std::filesystem::path summary_file_path_;
};

}  // namespace starpu_server
