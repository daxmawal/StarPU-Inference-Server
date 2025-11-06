#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <mutex>
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
      DeviceType worker_type = DeviceType::Unknown);
  void log_batch_completed(
      int batch_id, std::string_view model_name, std::size_t logical_jobs,
      std::size_t sample_count, int worker_id,
      DeviceType worker_type = DeviceType::Unknown);

 private:
  void write_record(
      BatchingTraceEvent event, std::string_view model_name, int request_id,
      int batch_id, std::size_t logical_jobs, std::size_t sample_count,
      int worker_id, DeviceType worker_type);
  [[nodiscard]] static auto event_to_string(BatchingTraceEvent event)
      -> std::string_view;
  [[nodiscard]] static auto device_type_to_string(DeviceType type)
      -> std::string_view;
  [[nodiscard]] static auto escape_json_string(std::string_view value)
      -> std::string;

  void write_header_locked();
  void write_footer_locked();
  void close_stream_locked();
  void write_line_locked(const std::string& line);
  void write_process_metadata_locked();
  void ensure_thread_metadata_locked(
      int thread_id, std::string_view thread_name, int sort_index);

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
};

}  // namespace starpu_server
