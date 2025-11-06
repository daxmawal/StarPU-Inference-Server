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

#include "device_type.hpp"

namespace starpu_server {

struct RuntimeConfig;

enum class BatchingTraceEvent : uint8_t {
  RequestQueued,
  RequestAssigned,
  BatchSubmitted,
  BatchCompleted
};

class BatchingTraceLogger {
 public:
  static auto instance() -> BatchingTraceLogger&;
  ~BatchingTraceLogger();

  void configure(bool enabled, std::string file_path);
  void configure_from_runtime(const RuntimeConfig& cfg);

  [[nodiscard]] auto enabled() const -> bool;

  void log_request_enqueued(int request_id, std::string_view model_name);
  void log_request_assigned_to_batch(
      int request_id, int batch_id, std::string_view model_name,
      std::size_t logical_jobs, std::size_t sample_count, int worker_id = -1,
      DeviceType worker_type = DeviceType::Unknown);
  void log_batch_submitted(
      int batch_id, std::string_view model_name, std::size_t logical_jobs,
      std::size_t sample_count, int worker_id = -1,
      DeviceType worker_type = DeviceType::Unknown,
      std::span<const int> request_ids = {});
  void log_batch_completed(
      int batch_id, std::string_view model_name, std::size_t logical_jobs,
      std::size_t sample_count, int worker_id,
      DeviceType worker_type = DeviceType::Unknown,
      std::chrono::high_resolution_clock::time_point codelet_start = {},
      std::chrono::high_resolution_clock::time_point codelet_end = {});

 private:
  void write_record(
      BatchingTraceEvent event, std::string_view model_name, int request_id,
      int batch_id, std::size_t logical_jobs, std::size_t sample_count,
      int worker_id, DeviceType worker_type, std::span<const int> request_ids,
      std::optional<int64_t> override_timestamp = std::nullopt,
      std::optional<int64_t> compute_start_ts = std::nullopt);
  void write_batch_compute_span(
      std::string_view model_name, int batch_id, std::size_t logical_jobs,
      std::size_t sample_count, int worker_id, DeviceType worker_type,
      int64_t start_ts, int64_t duration_us);
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
  void write_queue_track_marker_locked();
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
};

}  // namespace starpu_server
