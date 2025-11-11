#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
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
  void log_batch_submitted(
      int batch_id, std::string_view model_name,
      std::size_t logical_job_count = 0,
      DeviceType worker_type = DeviceType::Unknown, int worker_id = -1,
      std::span<const int> request_ids = {}, bool is_warmup = false,
      int device_id = -1);
  void log_batch_build_span(
      int batch_id, std::string_view model_name, std::size_t batch_size,
      TimeRange schedule, std::span<const int> request_ids = {},
      bool is_warmup = false);
  void log_batch_enqueue_span(
      int batch_id, std::string_view model_name, std::size_t batch_size,
      TimeRange queue_times, std::span<const int> request_ids = {},
      bool is_warmup = false);
  void log_batch_compute_span(
      int batch_id, std::string_view model_name, std::size_t batch_size,
      int worker_id, DeviceType worker_type = DeviceType::Unknown,
      TimeRange codelet_times = {}, bool is_warmup = false, int device_id = -1);

 private:
  void write_record(
      BatchingTraceEvent event, std::string_view model_name,
      const BatchRecordContext& record_context,
      const WorkerThreadInfo& worker_info, std::span<const int> request_ids,
      std::optional<int64_t> override_timestamp = std::nullopt,
      bool is_warmup = false);
  void write_batch_compute_span(
      std::string_view model_name, int batch_id, std::size_t batch_size,
      int worker_id, DeviceType worker_type, int64_t start_ts,
      int64_t duration_us, bool is_warmup, int device_id);
  void write_batch_enqueue_span(
      std::string_view model_name, int batch_id, std::size_t batch_size,
      BatchSpanTiming timing, std::span<const int> request_ids, bool is_warmup);
  void write_batch_build_span(
      std::string_view model_name, int batch_id, std::size_t batch_size,
      BatchSpanTiming timing, std::span<const int> request_ids, bool is_warmup);
  [[nodiscard]] static auto event_to_string(BatchingTraceEvent event)
      -> std::string_view;
  [[nodiscard]] static auto device_type_to_string(DeviceType type)
      -> std::string_view;
  [[nodiscard]] static auto escape_json_string(std::string_view value)
      -> std::string;
  [[nodiscard]] auto relative_timestamp_from_time_point(
      std::chrono::high_resolution_clock::time_point time_point) const
      -> std::optional<int64_t>;

  void write_header_locked();
  void write_footer_locked();
  void close_stream_locked();
  void write_line_locked(const std::string& line);
  void write_process_metadata_locked();
  void ensure_thread_metadata_locked(
      int thread_id, std::string_view thread_name, int sort_index);
  void remember_request_enqueue_timestamp(int request_id, int64_t timestamp_us);
  [[nodiscard]] auto consume_latest_request_enqueue_timestamp(
      std::span<const int> request_ids) -> std::optional<int64_t>;
  [[nodiscard]] static auto now_us() -> int64_t;
  [[nodiscard]] auto relative_timestamp_us(int64_t absolute_us) const
      -> int64_t;

  struct ThreadMetadata {
    std::string name;
    bool sort_emitted = false;
  };

  bool first_record_{true};
  bool header_written_{false};
  std::unordered_map<int, ThreadMetadata> thread_metadata_;
  std::mutex mutex_;
  std::ofstream stream_;
  std::atomic<bool> enabled_{false};
  std::string file_path_;
  int64_t trace_start_us_{0};
  bool trace_start_initialized_{false};
  struct WorkerLaneAssignment {
    int thread_id;
    int sort_index;
    int lane_index;
  };
  struct WorkerLaneKey {
    int worker_id;
    int lane_index;
  };
  struct WorkerLaneSpan {
    int64_t start_ts;
    int64_t end_ts;
  };
  struct WorkerLaneState {
    int thread_id;
    int64_t last_end_ts;
  };
  [[nodiscard]] auto assign_worker_lane_locked(
      int worker_id, WorkerLaneSpan lane_span) -> WorkerLaneAssignment;
  [[nodiscard]] static auto worker_lane_sort_index(WorkerLaneKey lane_key)
      -> int;
  [[nodiscard]] static auto worker_lane_thread_id(WorkerLaneKey lane_key)
      -> int;
  [[nodiscard]] static auto format_worker_lane_label(
      WorkerLaneKey lane_key, std::string_view worker_type_str,
      int device_id) -> std::string;

  std::unordered_map<int, std::vector<WorkerLaneState>> worker_lanes_;
  static constexpr int kWorkerLaneSortStride = 1000;
  std::mutex request_time_mutex_;
  std::unordered_map<int, int64_t> request_enqueue_times_;
};

}  // namespace starpu_server
