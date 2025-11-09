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
  static auto instance() -> BatchingTraceLogger&;
  ~BatchingTraceLogger();

  void configure(bool enabled, std::string file_path);
  void configure_from_runtime(const RuntimeConfig& cfg);

  [[nodiscard]] auto enabled() const -> bool;

  void log_request_enqueued(
      int request_id, std::string_view model_name, bool is_warmup = false,
      std::chrono::high_resolution_clock::time_point event_time = {});
  void log_batch_submitted(
      int batch_id, std::string_view model_name, std::size_t logical_jobs,
      int worker_id = -1, DeviceType worker_type = DeviceType::Unknown,
      std::span<const int> request_ids = {}, bool is_warmup = false,
      int device_id = -1);
  void log_batch_build_span(
      int batch_id, std::string_view model_name, std::size_t batch_size,
      std::chrono::high_resolution_clock::time_point start_time,
      std::chrono::high_resolution_clock::time_point end_time,
      std::span<const int> request_ids = {}, bool is_warmup = false);
  void log_batch_enqueue_span(
      int batch_id, std::string_view model_name, std::size_t batch_size,
      std::chrono::high_resolution_clock::time_point start_time,
      std::chrono::high_resolution_clock::time_point end_time,
      std::span<const int> request_ids = {}, bool is_warmup = false);
  void log_batch_compute_span(
      int batch_id, std::string_view model_name, std::size_t batch_size,
      int worker_id, DeviceType worker_type = DeviceType::Unknown,
      std::chrono::high_resolution_clock::time_point codelet_start = {},
      std::chrono::high_resolution_clock::time_point codelet_end = {},
      bool is_warmup = false, int device_id = -1);

 private:
  void write_record(
      BatchingTraceEvent event, std::string_view model_name, int request_id,
      int batch_id, std::size_t logical_jobs, int worker_id,
      DeviceType worker_type, std::span<const int> request_ids,
      std::optional<int64_t> override_timestamp = std::nullopt,
      bool is_warmup = false, int device_id = -1);
  void write_batch_compute_span(
      std::string_view model_name, int batch_id, std::size_t batch_size,
      int worker_id, DeviceType worker_type, int64_t start_ts,
      int64_t duration_us, bool is_warmup, int device_id);
  void write_batch_enqueue_span(
      std::string_view model_name, int batch_id, std::size_t batch_size,
      int64_t start_ts, int64_t duration_us, std::span<const int> request_ids,
      bool is_warmup);
  void write_batch_build_span(
      std::string_view model_name, int batch_id, std::size_t batch_size,
      int64_t start_ts, int64_t duration_us, std::span<const int> request_ids,
      bool is_warmup);
  [[nodiscard]] static auto event_to_string(BatchingTraceEvent event)
      -> std::string_view;
  [[nodiscard]] static auto device_type_to_string(DeviceType type)
      -> std::string_view;
  [[nodiscard]] static auto escape_json_string(std::string_view value)
      -> std::string;
  [[nodiscard]] auto relative_timestamp_from_time_point(
      std::chrono::high_resolution_clock::time_point tp) const
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
  [[nodiscard]] auto now_us() const -> int64_t;
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
  struct WorkerLaneState {
    int thread_id;
    int64_t last_end_ts;
  };
  [[nodiscard]] auto assign_worker_lane_locked(
      int worker_id, int64_t start_ts, int64_t end_ts) -> WorkerLaneAssignment;
  [[nodiscard]] static auto worker_lane_sort_index(
      int worker_id, int lane_index) -> int;
  [[nodiscard]] static auto format_worker_lane_label(
      int worker_id, std::string_view worker_type_str, int device_id,
      int lane_index) -> std::string;

  std::unordered_map<int, std::vector<WorkerLaneState>> worker_lanes_;
  static constexpr int kWorkerLaneSortStride = 1000;
  std::mutex request_time_mutex_;
  std::unordered_map<int, int64_t> request_enqueue_times_;
};

}  // namespace starpu_server
