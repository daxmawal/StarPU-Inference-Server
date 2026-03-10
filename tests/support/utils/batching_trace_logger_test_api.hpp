#pragma once

#if !defined(STARPU_TESTING)
#error \
    "batching_trace_logger_test_api.hpp is test-only and requires STARPU_TESTING"
#endif

#include <sstream>

#include "utils/batching_trace_logger.hpp"

namespace starpu_server::testing {

enum class FlowDirectionForTest : uint8_t {
  None = 0,
  Source = 1,
  Target = 2,
  SourceAndTarget = 3,
};

auto EscapeCsvFieldForTest(std::string_view value) -> std::string;
void AppendFlowAnnotationForTest(
    std::ostringstream& line, FlowDirectionForTest direction, int batch_id,
    bool is_warmup);

class TraceFileWriterTestAccessor {
 public:
  static auto stream(detail::TraceFileWriter& writer) -> std::ofstream&
  {
    return writer.stream_;
  }

  static auto header_written(detail::TraceFileWriter& writer) -> bool&
  {
    return writer.header_written_;
  }

  static auto first_record(detail::TraceFileWriter& writer) -> bool&
  {
    return writer.first_record_;
  }

  static auto thread_metadata(detail::TraceFileWriter& writer)
      -> decltype(writer.thread_metadata_)&
  {
    return writer.thread_metadata_;
  }
};

class BatchingTraceLoggerTestAccessor {
 public:
  static auto file_path(BatchingTraceLogger& logger) -> std::string&
  {
    return logger.file_path_;
  }

  static auto trace_writer(BatchingTraceLogger& logger)
      -> detail::TraceFileWriter&
  {
    return logger.trace_writer_;
  }

  static auto summary_stream(BatchingTraceLogger& logger) -> std::ofstream&
  {
    return logger.summary_writer_.summary_stream_;
  }

  static auto mutex(BatchingTraceLogger& logger) -> std::mutex&
  {
    return logger.mutex_;
  }

  static auto enabled_flag(BatchingTraceLogger& logger) -> std::atomic<bool>&
  {
    return logger.enabled_;
  }

  static auto request_timeline(BatchingTraceLogger& logger)
      -> detail::RequestTimelineTracker&
  {
    return logger.request_timeline_;
  }

  static auto summary_file_path(BatchingTraceLogger& logger)
      -> std::filesystem::path&
  {
    return logger.summary_writer_.summary_file_path_;
  }

  static auto queue_metrics_stream(BatchingTraceLogger& logger)
      -> std::ofstream&
  {
    return logger.summary_writer_.queue_metrics_stream_;
  }

  static auto queue_metrics_path(BatchingTraceLogger& logger)
      -> std::filesystem::path&
  {
    return logger.summary_writer_.queue_metrics_path_;
  }

  static auto rejected_total(BatchingTraceLogger& logger)
      -> std::atomic<std::size_t>&
  {
    return logger.rejected_total_;
  }

  static auto warmup_suppressed(BatchingTraceLogger& logger)
      -> std::atomic<bool>&
  {
    return logger.warmup_suppressed_;
  }

  static auto trace_start_us(BatchingTraceLogger& logger) -> int64_t&
  {
    return logger.trace_start_us_;
  }

  static auto trace_start_initialized(BatchingTraceLogger& logger) -> bool&
  {
    return logger.trace_start_initialized_;
  }

  static auto relative_timestamp_us(
      BatchingTraceLogger& logger, int64_t timestamp_us) -> int64_t
  {
    return logger.relative_timestamp_us(timestamp_us);
  }

  static auto configure_summary_writer(
      BatchingTraceLogger& logger,
      const std::filesystem::path& trace_path) -> bool
  {
    return logger.summary_writer_.open(trace_path);
  }

  static auto configure_queue_metrics_writer(
      BatchingTraceLogger& logger,
      const std::filesystem::path& trace_path) -> bool
  {
    return logger.summary_writer_.open_queue_metrics(trace_path);
  }

  static void write_batch_compute_span(
      BatchingTraceLogger& logger,
      const BatchingTraceLogger::BatchComputeWriteArgs& args)
  {
    logger.write_batch_compute_span(args);
  }

  static void write_batch_enqueue_span(
      BatchingTraceLogger& logger, int batch_id, std::string_view model_name,
      std::size_t batch_size, BatchingTraceLogger::BatchSpanTiming timing,
      std::span<const int> request_ids, bool is_warmup)
  {
    logger.write_batch_enqueue_span(
        batch_id, model_name, batch_size, timing, request_ids, is_warmup);
  }

  static void write_batch_build_span(
      BatchingTraceLogger& logger, int batch_id, std::string_view model_name,
      std::size_t batch_size, BatchingTraceLogger::BatchSpanTiming timing,
      std::span<const int> request_ids, bool is_warmup)
  {
    logger.write_batch_build_span(
        batch_id, model_name, batch_size, timing, request_ids, is_warmup);
  }

  static void write_record(
      BatchingTraceLogger& logger, BatchingTraceEvent event,
      std::string_view model_name,
      const BatchingTraceLogger::BatchRecordContext& record_context,
      const BatchingTraceLogger::WorkerThreadInfo& worker_info,
      std::span<const int> request_ids,
      std::optional<int64_t> override_timestamp, bool is_warmup)
  {
    logger.write_record(
        event, model_name, record_context, worker_info, request_ids,
        override_timestamp, is_warmup);
  }

  static void write_queue_metric_locked(
      BatchingTraceLogger& logger,
      const BatchingTraceLogger::QueueMetric& metric)
  {
    logger.summary_writer_.write_queue_metric_locked(metric);
  }
};

}  // namespace starpu_server::testing
