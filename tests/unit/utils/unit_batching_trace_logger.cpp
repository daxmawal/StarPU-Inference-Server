#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <span>
#include <sstream>
#include <string>

#include "utils/batching_trace_logger.hpp"
#include "utils/batching_trace_logger_test_api.hpp"
#include "utils/runtime_config.hpp"

namespace starpu_server { namespace {

auto
make_temp_trace_path() -> std::filesystem::path
{
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         std::format("perfetto_trace_test_{}.json", now);
}

auto
make_summary_path(const std::filesystem::path& trace_path)
    -> std::filesystem::path
{
  auto summary = trace_path;
  summary.replace_filename("trace.csv");
  return summary;
}

void
remove_trace_outputs(const std::filesystem::path& trace_path)
{
  std::error_code ec;
  std::filesystem::remove(trace_path, ec);
  std::filesystem::remove(make_summary_path(trace_path), ec);
  auto metrics = trace_path;
  metrics.replace_filename("metrics.csv");
  std::filesystem::remove(metrics, ec);
}

auto
read_summary_lines(const std::filesystem::path& summary_path)
    -> std::vector<std::string>
{
  std::ifstream stream(summary_path);
  if (!stream.is_open()) {
    return {};
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(stream, line)) {
    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

TEST(BatchingTraceLoggerTest, BatchSubmittedSkipsFlowMetadataForNegativeId)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  logger.log_batch_submitted(BatchingTraceLogger::BatchSubmittedLogArgs{
      .batch_id = -7,
      .model_name = "demo_model",
      .logical_job_count = 3,
  });
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  ASSERT_NE(content.find("\"batch_id\":-7"), std::string::npos);
  EXPECT_EQ(content.find("\"id_scope\""), std::string::npos);
  EXPECT_EQ(content.find("\"flow_out\":true"), std::string::npos);
  EXPECT_EQ(content.find("\"flow_in\":true"), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, RequestQueuedEventsEmitNoFlowAnnotations)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  logger.log_request_enqueued(9, "demo_model");
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  ASSERT_NE(content.find("\"request_id\":9"), std::string::npos);
  EXPECT_EQ(content.find("\"id_scope\""), std::string::npos);
  EXPECT_EQ(content.find("\"flow_out\":true"), std::string::npos);
  EXPECT_EQ(content.find("\"flow_in\":true"), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, RequestAndBatchLoggingNoOpsWhenDisabled)
{
  BatchingTraceLogger logger;
  const auto trace_path = make_temp_trace_path();
  remove_trace_outputs(trace_path);

  logger.log_request_enqueued(42, "demo_model");
  logger.log_batch_submitted(BatchingTraceLogger::BatchSubmittedLogArgs{
      .batch_id = 3,
      .model_name = "demo_model",
      .logical_job_count = 1,
  });

  EXPECT_FALSE(logger.enabled());
  EXPECT_TRUE(
      starpu_server::testing::BatchingTraceLoggerTestAccessor::file_path(logger)
          .empty());
  EXPECT_FALSE(
      starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
          logger)
          .is_open());
  EXPECT_FALSE(
      starpu_server::testing::BatchingTraceLoggerTestAccessor::summary_stream(
          logger)
          .is_open());
  EXPECT_FALSE(std::filesystem::exists(trace_path));
}

TEST(BatchingTraceLoggerTest, FlowAnnotationsIgnoreUnknownDirections)
{
  std::ostringstream line;
  line << R"({"prefix":true)";

  const auto invalid_direction =
      static_cast<starpu_server::testing::FlowDirectionForTest>(0xFF);
  starpu_server::testing::AppendFlowAnnotationForTest(
      line, invalid_direction, /*batch_id=*/42, /*is_warmup=*/false);

  EXPECT_EQ(line.str(), R"({"prefix":true)");
  EXPECT_EQ(line.str().find("\"flow_"), std::string::npos);
  EXPECT_EQ(line.str().find("\"id_scope\""), std::string::npos);
}

TEST(BatchingTraceLoggerTest, EscapeCsvFieldDoublesQuotes)
{
  const std::string_view input = R"(a"b"c)";
  EXPECT_EQ(starpu_server::testing::EscapeCsvFieldForTest(input), R"(a""b""c)");
}

TEST(BatchingTraceLoggerTest, ConfigureUsesDefaultPathWhenFilePathEmpty)
{
  BatchingTraceLogger logger;
  remove_trace_outputs(std::string(kDefaultTraceFileName));

  logger.configure(true, "");
  EXPECT_TRUE(logger.enabled());
  EXPECT_EQ(
      starpu_server::testing::BatchingTraceLoggerTestAccessor::file_path(
          logger),
      std::string(kDefaultTraceFileName));

  {
    std::ifstream stream{std::string(kDefaultTraceFileName)};
    EXPECT_TRUE(stream.is_open());
  }

  logger.configure(false, "");
  remove_trace_outputs(std::string(kDefaultTraceFileName));
}

TEST(BatchingTraceLoggerTest, ConfigureHandlesDirectoryCreationFailures)
{
  BatchingTraceLogger logger;
  const auto temp_dir =
      std::filesystem::temp_directory_path() /
      std::format(
          "batching_trace_conflict_{}",
          std::chrono::steady_clock::now().time_since_epoch().count());
  std::filesystem::create_directories(temp_dir);
  const auto conflicting_parent = temp_dir / "not_a_directory";
  {
    std::ofstream file(conflicting_parent);
    ASSERT_TRUE(file.is_open());
    file << "conflict";
  }
  const auto trace_path = conflicting_parent / "trace.json";

  logger.configure(true, trace_path.string());
  EXPECT_FALSE(logger.enabled());
  EXPECT_TRUE(
      starpu_server::testing::BatchingTraceLoggerTestAccessor::file_path(logger)
          .empty());
  EXPECT_FALSE(std::filesystem::exists(trace_path));

  std::error_code ec;
  std::filesystem::remove(conflicting_parent, ec);
  std::filesystem::remove(temp_dir, ec);
}

TEST(BatchingTraceLoggerTest, RelativeTimestampBypassesOffsetWhenUninitialized)
{
  BatchingTraceLogger fresh_logger;
  const int64_t sample_timestamp = 123456;

  EXPECT_EQ(
      starpu_server::testing::BatchingTraceLoggerTestAccessor::
          relative_timestamp_us(fresh_logger, sample_timestamp),
      sample_timestamp);

  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(fresh_logger) = true;
  starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_start_us(
      fresh_logger) = sample_timestamp + 50;
  EXPECT_EQ(
      starpu_server::testing::BatchingTraceLoggerTestAccessor::
          relative_timestamp_us(fresh_logger, sample_timestamp),
      0);
}

TEST(BatchingTraceLoggerTest, WriteLineLockedNoOpsWithoutHeader)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = false;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;

    starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
        logger)
        .write_line(R"({"dummy":"event"})");
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty())
      << "trace writer should not emit content before header.";

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, WriteFooterLockedNoOpsWithoutHeader)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = false;

    starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
        logger)
        .write_footer();
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty())
      << "trace writer should not emit content before header.";

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, WriteBatchBuildSpanClampsNonPositiveDuration)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;
  logger.configure(true, trace_path.string());

  BatchingTraceLogger::BatchSpanTiming timing{
      .start_ts = 5000,
      .duration_us = 0,
  };
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      write_batch_build_span(
          logger, "demo_model", 9, 3, timing, std::span<const int>{},
          /*is_warmup=*/false);
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  const std::string span_token = "\"name\":\"batch_build\"";
  const auto span_pos = content.find(span_token);
  ASSERT_NE(span_pos, std::string::npos);
  const std::string dur_token = "\"dur\":";
  const auto dur_pos = content.find(dur_token, span_pos);
  ASSERT_NE(dur_pos, std::string::npos);
  const auto value_start = dur_pos + dur_token.size();
  const auto value_end = content.find(',', value_start);
  ASSERT_NE(value_end, std::string::npos);
  const auto duration_value =
      content.substr(value_start, value_end - value_start);
  EXPECT_EQ(duration_value, "1");
  EXPECT_EQ(content.find("\"dur\":0", span_pos), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, WriteBatchEnqueueSpanClampsNonPositiveDuration)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;
  logger.configure(true, trace_path.string());

  BatchingTraceLogger::BatchSpanTiming timing{
      .start_ts = 2000,
      .duration_us = 0,
  };
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      write_batch_enqueue_span(
          logger, "demo_model", 7, 2, timing, std::span<const int>{},
          /*is_warmup=*/false);
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  const std::string span_token = "\"name\":\"batch\"";
  const auto span_pos = content.find(span_token);
  ASSERT_NE(span_pos, std::string::npos);
  const std::string dur_token = "\"dur\":";
  const auto dur_pos = content.find(dur_token, span_pos);
  ASSERT_NE(dur_pos, std::string::npos);
  const auto value_start = dur_pos + dur_token.size();
  const auto value_end = content.find(',', value_start);
  ASSERT_NE(value_end, std::string::npos);
  const auto duration_value =
      content.substr(value_start, value_end - value_start);
  EXPECT_EQ(duration_value, "1");
  EXPECT_EQ(content.find("\"dur\":0", span_pos), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, WriteBatchBuildSpanSkipsWithoutHeader)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = false;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
  }

  BatchingTraceLogger::BatchSpanTiming timing{
      .start_ts = 10,
      .duration_us = 5,
  };
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      write_batch_build_span(
          logger, "demo_model", 1, 1, timing, std::span<const int>{},
          /*is_warmup=*/false);

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty())
      << "write_batch_build_span should not emit without an open header.";

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, WriteBatchEnqueueSpanSkipsWithoutHeader)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = false;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
  }

  BatchingTraceLogger::BatchSpanTiming timing{
      .start_ts = 50,
      .duration_us = 10,
  };
  const std::array<int, 1> request_ids{42};
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      write_batch_enqueue_span(
          logger, "demo_model", 3, 1, timing, std::span<const int>(request_ids),
          /*is_warmup=*/false);

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty())
      << "write_batch_enqueue_span should not emit without an open header.";

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, LogBatchEnqueueSpanSkipsWhenDisabled)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
  }
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(logger) = true;
  starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_start_us(
      logger) = 0;

  const auto start = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{1000}};
  const auto end = start + std::chrono::microseconds{10};
  const BatchingTraceLogger::TimeRange queue_times{start, end};

  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(false, std::memory_order_release);
  logger.log_batch_enqueue_span(1, "disabled_model", 1, queue_times);

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty())
      << "log_batch_enqueue_span should not emit when tracing is disabled.";

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, LogBatchEnqueueSpanSkipsWithoutValidTimestamps)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
  }
  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(true, std::memory_order_release);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(logger) = true;
  starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_start_us(
      logger) = 0;

  const auto invalid_start = starpu_server::MonotonicClock::time_point{};
  const auto valid_end = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{5000}};
  const BatchingTraceLogger::TimeRange queue_times{invalid_start, valid_end};

  logger.log_batch_enqueue_span(2, "invalid_timestamps", 1, queue_times);

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty())
      << "log_batch_enqueue_span should not emit when timestamps cannot be "
         "converted.";

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, LogBatchEnqueueSpanClampsNegativeDuration)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
  }
  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(true, std::memory_order_release);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(logger) = true;
  starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_start_us(
      logger) = 0;

  const auto start = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{5000}};
  const auto end = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{3000}};
  const BatchingTraceLogger::TimeRange queue_times{start, end};

  logger.log_batch_enqueue_span(3, "clamped_enqueue", 1, queue_times);

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  const std::string span_token = "\"name\":\"batch\"";
  const auto span_pos = content.find(span_token);
  ASSERT_NE(span_pos, std::string::npos);
  const std::string dur_token = "\"dur\":";
  const auto dur_pos = content.find(dur_token, span_pos);
  ASSERT_NE(dur_pos, std::string::npos);
  const auto value_start = dur_pos + dur_token.size();
  const auto value_end = content.find(',', value_start);
  ASSERT_NE(value_end, std::string::npos);
  const auto duration_value =
      content.substr(value_start, value_end - value_start);
  EXPECT_EQ(duration_value, "1");
  EXPECT_EQ(content.find("\"dur\":0", span_pos), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, LogBatchBuildSpanSkipsWhenDisabled)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
  }
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(logger) = true;
  starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_start_us(
      logger) = 0;

  const auto start = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{1500}};
  const auto end = start + std::chrono::microseconds{20};
  const BatchingTraceLogger::TimeRange schedule{start, end};

  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(false, std::memory_order_release);
  logger.log_batch_build_span(4, "disabled_build", 1, schedule);

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty())
      << "log_batch_build_span should not emit when tracing is disabled.";

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, LogBatchBuildSpanSkipsInvalidTimestamps)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
  }
  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(true, std::memory_order_release);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(logger) = true;
  starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_start_us(
      logger) = 0;

  const auto start = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{7000}};
  const auto end = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{6500}};
  const BatchingTraceLogger::TimeRange schedule{start, end};

  logger.log_batch_build_span(5, "invalid_build_timing", 1, schedule);

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty())
      << "log_batch_build_span should not emit when timestamps are invalid.";

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, LogCongestionSpanSkipsWithoutValidTimestamps)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
  }
  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(true, std::memory_order_release);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(logger) = true;
  starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_start_us(
      logger) = 0;

  const auto valid_start = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{1000}};
  const auto valid_end = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{2000}};

  logger.log_congestion_span(BatchingTraceLogger::TimeRange{
      starpu_server::MonotonicClock::time_point{}, valid_end});
  logger.log_congestion_span(BatchingTraceLogger::TimeRange{
      valid_start, starpu_server::MonotonicClock::time_point{}});

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty())
      << "log_congestion_span should not emit when timestamps are invalid.";

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, LogCongestionSpanSkipsWhenEndBeforeStart)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
  }
  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(true, std::memory_order_release);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(logger) = true;
  starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_start_us(
      logger) = 0;

  const auto start = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{5000}};
  const auto end = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{4000}};

  logger.log_congestion_span(BatchingTraceLogger::TimeRange{start, end});

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty())
      << "log_congestion_span should not emit when end precedes start.";

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, LogCongestionSpanSkipsWhenWriterNotReady)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = false;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
  }
  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(true, std::memory_order_release);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(logger) = true;
  starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_start_us(
      logger) = 0;

  const auto start = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{2500}};
  const auto end = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{2600}};

  logger.log_congestion_span(BatchingTraceLogger::TimeRange{start, end});

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty())
      << "log_congestion_span should not emit when trace writer is not ready.";

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, LogCongestionSpanWritesWhenReady)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
  }
  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(true, std::memory_order_release);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(logger) = true;
  starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_start_us(
      logger) = 0;

  const auto start = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{8000}};
  const auto end = start + std::chrono::microseconds{25};

  logger.log_congestion_span(BatchingTraceLogger::TimeRange{start, end});

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_FALSE(content.empty());
  EXPECT_NE(content.find("\"cname\":\"bad\""), std::string::npos);
  EXPECT_NE(content.find("\"state\":\"congested\""), std::string::npos);
  EXPECT_NE(content.find("\"cat\":\"monitoring\""), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, WriteBatchComputeSpanSkipsNegativeWorker)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;
  logger.configure(true, trace_path.string());

  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      write_batch_compute_span(
          logger, BatchingTraceLogger::BatchComputeWriteArgs{
                      .model_name = "skip_model",
                      .batch_id = 11,
                      .batch_size = 4,
                      .worker_id = -1,
                      .worker_type = DeviceType::CPU,
                      .start_ts = 100,
                      .duration_us = 10,
                      .is_warmup = false,
                      .device_id = -1,
                  });
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_EQ(content.find("skip_model"), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, WriteBatchComputeSpanClampsNonPositiveDuration)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;
  logger.configure(true, trace_path.string());

  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      write_batch_compute_span(
          logger, BatchingTraceLogger::BatchComputeWriteArgs{
                      .model_name = "clamp_model",
                      .batch_id = 12,
                      .batch_size = 5,
                      .worker_id = 2,
                      .worker_type = DeviceType::CPU,
                      .start_ts = 500,
                      .duration_us = 0,
                      .is_warmup = false,
                      .device_id = -1,
                  });
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  const std::string event_token = "\"name\":\"clamp_model\"";
  const auto event_pos = content.find(event_token);
  ASSERT_NE(event_pos, std::string::npos);
  const std::string dur_token = "\"dur\":";
  const auto dur_pos = content.find(dur_token, event_pos);
  ASSERT_NE(dur_pos, std::string::npos);
  const auto value_start = dur_pos + dur_token.size();
  const auto value_end = content.find(',', value_start);
  ASSERT_NE(value_end, std::string::npos);
  const auto duration_value =
      content.substr(value_start, value_end - value_start);
  EXPECT_EQ(duration_value, "1");
  EXPECT_EQ(content.find("\"dur\":0", event_pos), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, WriteBatchComputeSpanSkipsWithoutHeader)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = false;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
  }

  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      write_batch_compute_span(
          logger, BatchingTraceLogger::BatchComputeWriteArgs{
                      .model_name = "demo_model",
                      .batch_id = 13,
                      .batch_size = 2,
                      .worker_id = 3,
                      .worker_type = DeviceType::CPU,
                      .start_ts = 1000,
                      .duration_us = 25,
                      .is_warmup = false,
                      .device_id = -1,
                  });

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty())
      << "write_batch_compute_span should not emit without an open header.";

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, EventToStringReturnsUnknownForInvalidEvent)
{
  const auto value =
      detail::event_to_string(static_cast<BatchingTraceEvent>(255));
  EXPECT_EQ(value, "unknown");
}

TEST(BatchingTraceLoggerTest, RememberRequestEnqueueTimestampSkipsNegativeId)
{
  BatchingTraceLogger logger;
  starpu_server::testing::BatchingTraceLoggerTestAccessor::request_timeline(
      logger)
      .remember(-5, 1234);

  const std::array<int, 1> request_ids{-5};
  const auto timestamp =
      starpu_server::testing::BatchingTraceLoggerTestAccessor::request_timeline(
          logger)
          .consume_latest(request_ids);
  EXPECT_FALSE(timestamp.has_value())
      << "Negative request IDs should be ignored.";
}

TEST(BatchingTraceLoggerTest, WriteRecordSkipsWhenDisabled)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
  }
  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(false, std::memory_order_release);

  const BatchingTraceLogger::BatchRecordContext record_context{
      .request_id = 1, .batch_id = 1, .logical_jobs = 1};
  const BatchingTraceLogger::WorkerThreadInfo worker_info{
      .worker_id = -1, .worker_type = DeviceType::Unknown, .device_id = -1};

  starpu_server::testing::BatchingTraceLoggerTestAccessor::write_record(
      logger, BatchingTraceEvent::RequestQueued, "demo_model", record_context,
      worker_info, std::span<const int>{}, std::nullopt, /*is_warmup=*/false);

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty())
      << "write_record should not emit when tracing is disabled.";

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, WriteRecordSkipsWithoutHeader)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = false;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
  }
  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(true, std::memory_order_release);

  const BatchingTraceLogger::BatchRecordContext record_context{
      .request_id = 2, .batch_id = 3, .logical_jobs = 4};
  const BatchingTraceLogger::WorkerThreadInfo worker_info{
      .worker_id = -1, .worker_type = DeviceType::Unknown, .device_id = -1};

  starpu_server::testing::BatchingTraceLoggerTestAccessor::write_record(
      logger, BatchingTraceEvent::BatchSubmitted, "demo_model", record_context,
      worker_info, std::span<const int>{}, std::nullopt, /*is_warmup=*/false);

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty())
      << "write_record should not emit before the trace header is written.";

  remove_trace_outputs(trace_path);
}

TEST(
    BatchingTraceLoggerTest,
    LogBatchComputeSpanSkipsWhenDisabledOrWorkerInvalid)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
  }
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(logger) = true;
  starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_start_us(
      logger) = 0;

  const auto start = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{1000}};
  const auto end = start + std::chrono::microseconds{10};
  const BatchingTraceLogger::TimeRange codelet_times{start, end};

  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(false, std::memory_order_release);
  logger.log_batch_compute_span(BatchingTraceLogger::BatchComputeLogArgs{
      .batch_id = 1,
      .model_name = "disabled_model",
      .batch_size = 1,
      .worker_id = 5,
      .worker_type = DeviceType::CPU,
      .codelet_times = codelet_times,
      .is_warmup = false,
      .device_id = -1,
  });

  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(true, std::memory_order_release);
  logger.log_batch_compute_span(BatchingTraceLogger::BatchComputeLogArgs{
      .batch_id = 2,
      .model_name = "invalid_worker_model",
      .batch_size = 1,
      .worker_id = -1,
      .worker_type = DeviceType::CPU,
      .codelet_times = codelet_times,
      .is_warmup = false,
      .device_id = -1,
  });

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty()) << "log_batch_compute_span should not emit when "
                                  "disabled or worker_id < 0.";

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, LogBatchComputeSpanSkipsInvalidTimestamps)
{
  const auto trace_path = make_temp_trace_path();
  BatchingTraceLogger logger;

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .open(trace_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(starpu_server::testing::TraceFileWriterTestAccessor::stream(
                    starpu_server::testing::BatchingTraceLoggerTestAccessor::
                        trace_writer(logger))
                    .is_open());
    starpu_server::testing::TraceFileWriterTestAccessor::header_written(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
    starpu_server::testing::TraceFileWriterTestAccessor::first_record(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger)) = true;
  }
  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(true, std::memory_order_release);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(logger) = true;
  starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_start_us(
      logger) = 0;

  const auto start = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{2000}};
  const auto end = starpu_server::MonotonicClock::time_point{
      std::chrono::microseconds{1500}};
  const BatchingTraceLogger::TimeRange codelet_times{start, end};

  logger.log_batch_compute_span(BatchingTraceLogger::BatchComputeLogArgs{
      .batch_id = 3,
      .model_name = "invalid_time_model",
      .batch_size = 1,
      .worker_id = 7,
      .worker_type = DeviceType::CPU,
      .codelet_times = codelet_times,
      .is_warmup = false,
      .device_id = -1,
  });

  {
    std::lock_guard<std::mutex> lock(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::mutex(logger));
    starpu_server::testing::TraceFileWriterTestAccessor::stream(
        starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
            logger))
        .close();
  }

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty())
      << "log_batch_compute_span should not emit when timestamps are invalid.";

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, EscapesModelNamesWithSpecialCharacters)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const auto start = starpu_server::MonotonicClock::now();
  const auto end = start + std::chrono::microseconds(10);
  std::string model_name = "\"\\\b\f\n\r\tA";
  model_name.push_back(static_cast<char>(0x01));
  logger.log_batch_build_span(
      11, model_name, 1, BatchingTraceLogger::TimeRange{start, end},
      std::span<const int>{});
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());

  const std::string expected_fragment =
      "\"model_name\":\"\\\"\\\\\\b\\f\\n\\r\\tA\\u0001\"";
  EXPECT_NE(content.find(expected_fragment), std::string::npos)
      << "Escaped model name was not serialized as expected. Trace content:\n"
      << content;

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, RoutesNonWorkerEventsToDedicatedTracks)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  logger.log_request_enqueued(1, "demo_model");
  const auto enqueue_start = starpu_server::MonotonicClock::now();
  const auto enqueue_end = enqueue_start + std::chrono::microseconds(10);
  const std::array<int, 2> request_ids{0, 1};
  logger.log_batch_enqueue_span(
      5, "demo_model", request_ids.size(),
      BatchingTraceLogger::TimeRange{enqueue_start, enqueue_end},
      std::span<const int>(request_ids));
  logger.log_batch_submitted(BatchingTraceLogger::BatchSubmittedLogArgs{
      .batch_id = 5,
      .model_name = "demo_model",
      .logical_job_count = 1,
  });
  logger.log_batch_submitted(BatchingTraceLogger::BatchSubmittedLogArgs{
      .batch_id = 7,
      .model_name = "demo_model",
      .logical_job_count = 1,
      .worker_type = DeviceType::CPU,
      .worker_id = 0,
  });
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());

  EXPECT_EQ(content.find("\"worker_id\":-1"), std::string::npos);
  EXPECT_NE(content.find("\"tid\":1"), std::string::npos);
  EXPECT_NE(content.find("\"tid\":2"), std::string::npos);
  EXPECT_NE(content.find("\"tid\":3"), std::string::npos);
  EXPECT_NE(content.find("\"tid\":4"), std::string::npos);
  EXPECT_NE(content.find("request_enqueued"), std::string::npos);
  EXPECT_NE(content.find("\"name\":\"batch\""), std::string::npos);
  EXPECT_NE(content.find("batch_submitted"), std::string::npos);
  EXPECT_NE(content.find("\"batch_size\":1"), std::string::npos);
  EXPECT_NE(content.find("\"request_id\":1"), std::string::npos);
  EXPECT_EQ(content.find("\"batch_id\":-1"), std::string::npos);
  EXPECT_EQ(content.find("\"logical_jobs\":0"), std::string::npos);
  EXPECT_EQ(content.find("\"sample_count\":0"), std::string::npos);
  EXPECT_EQ(content.find("\"worker_type\":\"unknown\""), std::string::npos);
  EXPECT_EQ(content.find("task_queue"), std::string::npos);
  EXPECT_EQ(content.find("\"worker_id\":0"), std::string::npos);
  EXPECT_NE(content.find("\"tid\":10"), std::string::npos);
  EXPECT_NE(content.find("worker-0 (cpu)"), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, IncludesDeviceIdInWorkerLabels)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const auto start = starpu_server::MonotonicClock::now();
  const auto end = start + std::chrono::microseconds(50);
  logger.log_batch_submitted(BatchingTraceLogger::BatchSubmittedLogArgs{
      .batch_id = 3,
      .model_name = "demo_model",
      .logical_job_count = 1,
      .worker_type = DeviceType::CUDA,
      .worker_id = 4,
      .device_id = 7,
  });
  logger.log_batch_compute_span(BatchingTraceLogger::BatchComputeLogArgs{
      .batch_id = 3,
      .model_name = "demo_model",
      .batch_size = 1,
      .worker_id = 4,
      .worker_type = DeviceType::CUDA,
      .codelet_times = BatchingTraceLogger::TimeRange{start, end},
      .is_warmup = false,
      .device_id = 7,
  });
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());

  EXPECT_NE(content.find("device 7 worker 4 (cuda)"), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, RecordsRequestIdsForBatchSubmission)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const std::array<int, 3> request_ids{42, 43, 44};
  logger.log_batch_submitted(BatchingTraceLogger::BatchSubmittedLogArgs{
      .batch_id = 9,
      .model_name = "demo_model",
      .logical_job_count = 2,
      .worker_type = DeviceType::CUDA,
      .worker_id = 1,
      .request_ids = std::span<const int>(request_ids),
  });
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());

  EXPECT_NE(content.find("\"request_ids\":[42,43,44]"), std::string::npos);
  EXPECT_NE(content.find("\"batch_size\":2"), std::string::npos);
  EXPECT_EQ(content.find("\"worker_id\":1"), std::string::npos);
  EXPECT_EQ(content.find("\"worker_type\":\"cuda\""), std::string::npos);
  EXPECT_EQ(content.find("\"sample_count\":6"), std::string::npos);
  EXPECT_EQ(content.find("\"request_id\":"), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, EmitsBatchBuildSpanWithRequestIds)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const auto start = starpu_server::MonotonicClock::now();
  const auto end = start + std::chrono::microseconds(150);
  const std::array<int, 2> request_ids{7, 8};
  logger.log_batch_build_span(
      21, "demo_model", 9, BatchingTraceLogger::TimeRange{start, end},
      std::span<const int>(request_ids));
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());

  EXPECT_NE(content.find("\"name\":\"batch_build\""), std::string::npos);
  EXPECT_NE(content.find("\"tid\":2"), std::string::npos);
  EXPECT_NE(content.find("\"request_ids\":[7,8]"), std::string::npos);
  EXPECT_NE(content.find("\"batch_size\":9"), std::string::npos);
  EXPECT_EQ(content.find("\"logical_jobs\":"), std::string::npos);
  EXPECT_EQ(content.find("\"sample_count\":"), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, EmitsBatchEnqueueSpanWithRequestIds)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const auto start = starpu_server::MonotonicClock::now();
  const auto end = start + std::chrono::microseconds(60);
  const std::array<int, 3> request_ids{4, 5, 6};
  logger.log_batch_enqueue_span(
      21, "demo_model", request_ids.size(),
      BatchingTraceLogger::TimeRange{start, end},
      std::span<const int>(request_ids));
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());

  EXPECT_NE(content.find("\"name\":\"batch\""), std::string::npos);
  EXPECT_NE(content.find("\"tid\":4"), std::string::npos);
  EXPECT_NE(content.find("\"request_ids\":[4,5,6]"), std::string::npos);
  EXPECT_NE(content.find("\"batch_size\":3"), std::string::npos);
  EXPECT_NE(content.find("\"start_ts\":"), std::string::npos);
  EXPECT_NE(content.find("\"end_ts\":"), std::string::npos);
  EXPECT_EQ(content.find("\"logical_jobs\":"), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, ExtendsEnqueueWindowToLatestRequestEvent)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const auto base = starpu_server::MonotonicClock::now();
  const auto first = base + std::chrono::microseconds(10);
  const auto second = first + std::chrono::microseconds(5);
  const auto last = first + std::chrono::microseconds(25);
  logger.log_request_enqueued(101, "demo_model", false, first);
  logger.log_request_enqueued(102, "demo_model", false, second);
  logger.log_request_enqueued(103, "demo_model", false, last);
  const std::array<int, 3> request_ids{101, 102, 103};
  logger.log_batch_enqueue_span(
      99, "demo_model", request_ids.size(),
      BatchingTraceLogger::TimeRange{first, second},
      std::span<const int>(request_ids));
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());

  const auto window_pos = content.find("\"name\":\"batch\"");
  ASSERT_NE(window_pos, std::string::npos);
  const auto dur_pos = content.find("\"dur\":", window_pos);
  ASSERT_NE(dur_pos, std::string::npos);
  const auto value_start = dur_pos + std::string("\"dur\":").size();
  const auto value_end = content.find_first_of(",}", value_start);
  ASSERT_NE(value_end, std::string::npos);
  const auto duration =
      std::stoll(content.substr(value_start, value_end - value_start));
  EXPECT_EQ(duration, 25);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, PrefixesWarmupEvents)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const auto start = starpu_server::MonotonicClock::now();
  const auto end = start + std::chrono::microseconds(100);
  const std::array<int, 2> request_ids{1, 2};
  logger.log_batch_submitted(BatchingTraceLogger::BatchSubmittedLogArgs{
      .batch_id = 11,
      .model_name = "demo_model",
      .logical_job_count = 1,
      .worker_type = DeviceType::CPU,
      .worker_id = 0,
      .request_ids = std::span<const int>(request_ids),
      .is_warmup = true,
  });
  logger.log_batch_build_span(
      11, "demo_model", 1, BatchingTraceLogger::TimeRange{start, end},
      std::span<const int>(request_ids), /*is_warmup=*/true);
  logger.log_batch_compute_span(BatchingTraceLogger::BatchComputeLogArgs{
      .batch_id = 11,
      .model_name = "demo_model",
      .batch_size = 1,
      .worker_id = 0,
      .worker_type = DeviceType::CPU,
      .codelet_times = BatchingTraceLogger::TimeRange{start, end},
      .is_warmup = true,
  });
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());

  EXPECT_NE(
      content.find("\"name\":\"warming_batch_submitted\""), std::string::npos);
  EXPECT_NE(content.find("\"warming_batch_size\":1"), std::string::npos);
  EXPECT_NE(content.find("\"warming_request_ids\":[1,2]"), std::string::npos);
  EXPECT_NE(
      content.find("\"name\":\"warming_batch_build\""), std::string::npos);
  EXPECT_NE(content.find("\"warming_batch_id\""), std::string::npos);
  EXPECT_NE(content.find("\"name\":\"warming_demo_model\""), std::string::npos);
  EXPECT_NE(
      content.find(
          "\"warming_batch_size\":1,\"warming_model_name\":\"demo_model\","
          "\"warming_worker_id\""),
      std::string::npos);
  EXPECT_EQ(content.find("\"warming_sample_count\""), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, SplitsOverlappingComputeSpansIntoWorkerLanes)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const auto base = starpu_server::MonotonicClock::now();
  const auto overlapping_start = base + std::chrono::microseconds(50);
  const auto overlapping_end =
      overlapping_start + std::chrono::microseconds(40);
  const auto long_end = base + std::chrono::microseconds(180);

  logger.log_batch_compute_span(BatchingTraceLogger::BatchComputeLogArgs{
      .batch_id = 1,
      .model_name = "demo_model",
      .batch_size = 1,
      .worker_id = 0,
      .worker_type = DeviceType::CPU,
      .codelet_times =
          BatchingTraceLogger::TimeRange{overlapping_start, overlapping_end},
  });
  logger.log_batch_compute_span(BatchingTraceLogger::BatchComputeLogArgs{
      .batch_id = 2,
      .model_name = "demo_model",
      .batch_size = 1,
      .worker_id = 0,
      .worker_type = DeviceType::CPU,
      .codelet_times = BatchingTraceLogger::TimeRange{base, long_end},
  });
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());

  auto extract_tid = [&](size_t start_pos) -> int {
    const auto tid_pos = content.find("\"tid\":", start_pos);
    if (tid_pos == std::string::npos) {
      ADD_FAILURE() << "Missing tid for compute span";
      return -1;
    }
    const auto tid_end = content.find_first_of(",}", tid_pos);
    if (tid_end == std::string::npos) {
      ADD_FAILURE() << "Malformed tid for compute span";
      return -1;
    }
    const auto tid_str = content.substr(tid_pos + 6, tid_end - (tid_pos + 6));
    return std::stoi(tid_str);
  };

  const auto first_compute = content.find("\"name\":\"demo_model\"");
  ASSERT_NE(first_compute, std::string::npos);
  const auto second_compute =
      content.find("\"name\":\"demo_model\"", first_compute + 1);
  ASSERT_NE(second_compute, std::string::npos);

  const int tid_one = extract_tid(first_compute);
  const int tid_two = extract_tid(second_compute);
  EXPECT_NE(tid_one, tid_two);
  EXPECT_NE(content.find("worker-0 (cpu) #2"), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, EmitsScopedFlowsBetweenSubmissionAndCompute)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const auto start = starpu_server::MonotonicClock::now();
  const auto end = start + std::chrono::microseconds(75);
  const auto build_start = start - std::chrono::microseconds(50);
  const auto build_end = build_start + std::chrono::microseconds(20);
  const auto warm_build_start = end + std::chrono::microseconds(50);
  const auto warm_build_end = warm_build_start + std::chrono::microseconds(20);
  const auto warm_start = warm_build_end + std::chrono::microseconds(25);
  const auto warm_end = warm_start + std::chrono::microseconds(75);
  logger.log_batch_build_span(
      0, "demo_model", 2,
      BatchingTraceLogger::TimeRange{build_start, build_end},
      std::span<const int>{});
  logger.log_batch_submitted(BatchingTraceLogger::BatchSubmittedLogArgs{
      .batch_id = 0,
      .model_name = "demo_model",
      .logical_job_count = 2,
      .worker_type = DeviceType::CPU,
      .worker_id = 0,
  });
  logger.log_batch_compute_span(BatchingTraceLogger::BatchComputeLogArgs{
      .batch_id = 0,
      .model_name = "demo_model",
      .batch_size = 2,
      .worker_id = 0,
      .worker_type = DeviceType::CPU,
      .codelet_times = BatchingTraceLogger::TimeRange{start, end},
  });
  logger.log_batch_build_span(
      0, "demo_model", 1,
      BatchingTraceLogger::TimeRange{warm_build_start, warm_build_end},
      std::span<const int>{}, /*is_warmup=*/true);
  logger.log_batch_submitted(BatchingTraceLogger::BatchSubmittedLogArgs{
      .batch_id = 0,
      .model_name = "demo_model",
      .logical_job_count = 1,
      .worker_type = DeviceType::CPU,
      .worker_id = 1,
      .is_warmup = true,
  });
  logger.log_batch_compute_span(BatchingTraceLogger::BatchComputeLogArgs{
      .batch_id = 0,
      .model_name = "demo_model",
      .batch_size = 1,
      .worker_id = 1,
      .worker_type = DeviceType::CPU,
      .codelet_times = BatchingTraceLogger::TimeRange{warm_start, warm_end},
      .is_warmup = true,
  });
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  const auto serving_compute = content.find("\"name\":\"demo_model\"");
  ASSERT_NE(serving_compute, std::string::npos);
  const auto warming_compute = content.find("\"name\":\"warming_demo_model\"");
  ASSERT_NE(warming_compute, std::string::npos);

  const auto serving_build =
      content.find("\"name\":\"batch_build\",\"cat\":\"batching\"");
  ASSERT_NE(serving_build, std::string::npos);
  const auto serving_submit =
      content.find("\"name\":\"batch_submitted\",\"cat\":\"batching\"");
  ASSERT_NE(serving_submit, std::string::npos);
  EXPECT_LT(serving_build, serving_submit);
  EXPECT_NE(
      content.find(
          "\"id_scope\":\"serving\",\"id2\":{\"local\":0}", serving_build),
      std::string::npos);
  const auto serving_build_flow_out =
      content.find("\"flow_out\":true", serving_build);
  ASSERT_NE(serving_build_flow_out, std::string::npos);
  EXPECT_LT(serving_build_flow_out, serving_submit);
  const auto serving_build_bind =
      content.find("\"bind_id\":\"0x0000000000000000\"", serving_build);
  ASSERT_NE(serving_build_bind, std::string::npos);
  EXPECT_LT(serving_build_bind, serving_submit);
  EXPECT_NE(
      content.find(
          "\"id_scope\":\"serving\",\"id2\":{\"local\":0}", serving_submit),
      std::string::npos);
  const auto serving_submit_flow_in =
      content.find("\"flow_in\":true", serving_submit);
  ASSERT_NE(serving_submit_flow_in, std::string::npos);
  EXPECT_LT(serving_submit_flow_in, serving_compute);
  const auto serving_submit_flow_out =
      content.find("\"flow_out\":true", serving_submit);
  ASSERT_NE(serving_submit_flow_out, std::string::npos);
  EXPECT_LT(serving_submit_flow_out, serving_compute);
  EXPECT_NE(
      content.find("\"bind_id\":\"0x0000000000000000\"", serving_submit),
      std::string::npos);

  const auto warming_submit =
      content.find("\"name\":\"warming_batch_submitted\"");
  const auto warming_build = content.find("\"name\":\"warming_batch_build\"");
  ASSERT_NE(warming_build, std::string::npos);
  ASSERT_NE(warming_submit, std::string::npos);
  EXPECT_LT(warming_build, warming_submit);
  EXPECT_NE(
      content.find(
          "\"id_scope\":\"warming\",\"id2\":{\"local\":0}", warming_build),
      std::string::npos);
  const auto warming_build_flow_out =
      content.find("\"flow_out\":true", warming_build);
  ASSERT_NE(warming_build_flow_out, std::string::npos);
  EXPECT_LT(warming_build_flow_out, warming_submit);
  const auto warming_build_bind =
      content.find("\"bind_id\":\"0x8000000000000000\"", warming_build);
  ASSERT_NE(warming_build_bind, std::string::npos);
  EXPECT_LT(warming_build_bind, warming_submit);
  EXPECT_NE(
      content.find(
          "\"id_scope\":\"warming\",\"id2\":{\"local\":0}", warming_submit),
      std::string::npos);
  const auto warming_submit_flow_in =
      content.find("\"flow_in\":true", warming_submit);
  ASSERT_NE(warming_submit_flow_in, std::string::npos);
  EXPECT_LT(warming_submit_flow_in, warming_compute);
  const auto warming_submit_flow_out =
      content.find("\"flow_out\":true", warming_submit);
  ASSERT_NE(warming_submit_flow_out, std::string::npos);
  EXPECT_LT(warming_submit_flow_out, warming_compute);
  EXPECT_NE(
      content.find("\"bind_id\":\"0x8000000000000000\"", warming_submit),
      std::string::npos);

  EXPECT_NE(
      content.find("\"flow_in\":true", serving_compute), std::string::npos);
  EXPECT_NE(
      content.find("\"bind_id\":\"0x0000000000000000\"", serving_compute),
      std::string::npos);

  EXPECT_NE(
      content.find("\"flow_in\":true", warming_compute), std::string::npos);
  EXPECT_NE(
      content.find("\"bind_id\":\"0x8000000000000000\"", warming_compute),
      std::string::npos);

  EXPECT_EQ(content.find("\"ph\":\"s\""), std::string::npos);
  EXPECT_EQ(content.find("\"ph\":\"f\""), std::string::npos);
  EXPECT_EQ(content.find(",\"id\":"), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, WritesBatchSummaryEntries)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const std::array<int, 2> request_ids{100, 101};
  const std::array<int64_t, 2> request_arrivals{1000, 2000};
  logger.log_batch_summary(BatchingTraceLogger::BatchSummaryLogArgs{
      .batch_id = 99,
      .model_name = "demo_model",
      .batch_size = 3,
      .request_ids = request_ids,
      .request_arrival_us = request_arrivals,
      .worker_id = 7,
      .worker_type = DeviceType::CUDA,
      .device_id = 4,
      .queue_ms = 1.0,
      .batch_ms = 2.0,
      .submit_ms = 3.0,
      .scheduling_ms = 4.0,
      .codelet_ms = 5.0,
      .inference_ms = 6.0,
      .callback_ms = 7.0,
      .total_ms = 8.0,
      .is_warmup = false,
  });
  logger.configure(false, "");

  const auto summary_path = make_summary_path(trace_path);
  const auto lines = read_summary_lines(summary_path);
  ASSERT_EQ(lines.size(), 2U);
  EXPECT_TRUE(lines[0].starts_with("batch_id,model_name,worker_id"))
      << "Unexpected header line: " << lines[0];
  EXPECT_NE(lines[0].find("request_arrival_us"), std::string::npos)
      << "Missing arrival column header: " << lines[0];
  EXPECT_NE(lines[1].find("\"demo_model\""), std::string::npos);
  EXPECT_NE(lines[1].find("\"cuda\""), std::string::npos);
  EXPECT_NE(lines[1].find("\"100;101\""), std::string::npos);
  EXPECT_NE(lines[1].find("\"1000;2000\""), std::string::npos);
  EXPECT_NE(lines[1].find(",7.000,"), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, ConfigureSummaryWriterFailsWhenDirectoryMissing)
{
  BatchingTraceLogger logger;
  const auto trace_path = std::filesystem::temp_directory_path() /
                          "batching_trace_missing_dir" / "trace.json";
  std::error_code ec;
  std::filesystem::remove_all(trace_path.parent_path(), ec);

  const auto configured =
      starpu_server::testing::BatchingTraceLoggerTestAccessor::
          configure_summary_writer(logger, trace_path);
  EXPECT_FALSE(configured);
  EXPECT_TRUE(starpu_server::testing::BatchingTraceLoggerTestAccessor::
                  summary_file_path(logger)
                      .empty());
  EXPECT_FALSE(
      starpu_server::testing::BatchingTraceLoggerTestAccessor::summary_stream(
          logger)
          .is_open());
}

TEST(
    BatchingTraceLoggerTest,
    ConfigureQueueMetricsWriterFailsWhenDirectoryMissing)
{
  BatchingTraceLogger logger;
  const auto trace_path = std::filesystem::temp_directory_path() /
                          "batching_trace_missing_metrics_dir" / "trace.json";
  std::error_code ec;
  std::filesystem::remove_all(trace_path.parent_path(), ec);

  const auto configured =
      starpu_server::testing::BatchingTraceLoggerTestAccessor::
          configure_queue_metrics_writer(logger, trace_path);
  EXPECT_FALSE(configured);
  EXPECT_TRUE(starpu_server::testing::BatchingTraceLoggerTestAccessor::
                  queue_metrics_path(logger)
                      .empty());
  EXPECT_FALSE(starpu_server::testing::BatchingTraceLoggerTestAccessor::
                   queue_metrics_stream(logger)
                       .is_open());
}

TEST(BatchingTraceLoggerTest, BatchSummaryNoOpsWhenSummaryStreamClosed)
{
  BatchingTraceLogger logger;
  const auto trace_path = make_temp_trace_path();
  remove_trace_outputs(trace_path);
  const auto summary_path = make_summary_path(trace_path);

  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(true, std::memory_order_release);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::warmup_suppressed(
      logger)
      .store(false, std::memory_order_release);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::summary_file_path(
      logger) = summary_path;

  const std::array<int, 1> request_ids{1};
  const std::array<int64_t, 1> request_arrivals{100};
  logger.log_batch_summary(BatchingTraceLogger::BatchSummaryLogArgs{
      .batch_id = 1,
      .model_name = "demo_model",
      .batch_size = 1,
      .request_ids = request_ids,
      .request_arrival_us = request_arrivals,
      .worker_id = 0,
      .worker_type = DeviceType::CPU,
      .device_id = -1,
      .queue_ms = 0.1,
      .batch_ms = 0.2,
      .submit_ms = 0.3,
      .scheduling_ms = 0.4,
      .codelet_ms = 0.5,
      .inference_ms = 0.6,
      .callback_ms = 0.7,
      .total_ms = 0.8,
      .is_warmup = false,
  });

  EXPECT_FALSE(
      starpu_server::testing::BatchingTraceLoggerTestAccessor::summary_stream(
          logger)
          .is_open());
  EXPECT_FALSE(std::filesystem::exists(summary_path));
}

TEST(BatchingTraceLoggerTest, QueueMetricsNoOpsWhenNotReady)
{
  BatchingTraceLogger logger;
  const auto trace_path = make_temp_trace_path();
  remove_trace_outputs(trace_path);

  auto metrics_path = trace_path;
  metrics_path.replace_filename("metrics.csv");

  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(true, std::memory_order_release);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::warmup_suppressed(
      logger)
      .store(false, std::memory_order_release);

  starpu_server::testing::BatchingTraceLoggerTestAccessor::queue_metrics_stream(
      logger)
      .open(metrics_path, std::ios::out | std::ios::trunc);
  ASSERT_TRUE(starpu_server::testing::BatchingTraceLoggerTestAccessor::
                  queue_metrics_stream(logger)
                      .is_open());
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(logger) = false;
  logger.log_queue_size(3);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::queue_metrics_stream(
      logger)
      .close();

  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(logger) = true;
  logger.log_queue_size(4);

  std::ifstream stream(metrics_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty());

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, RequestRejectedNoOpsWhenMetricsNotReady)
{
  BatchingTraceLogger logger;
  const auto trace_path = make_temp_trace_path();
  remove_trace_outputs(trace_path);

  auto metrics_path = trace_path;
  metrics_path.replace_filename("metrics.csv");

  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(true, std::memory_order_release);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::warmup_suppressed(
      logger)
      .store(false, std::memory_order_release);

  starpu_server::testing::BatchingTraceLoggerTestAccessor::queue_metrics_stream(
      logger)
      .open(metrics_path, std::ios::out | std::ios::trunc);
  ASSERT_TRUE(starpu_server::testing::BatchingTraceLoggerTestAccessor::
                  queue_metrics_stream(logger)
                      .is_open());
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(logger) = false;
  logger.log_request_rejected(5);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::queue_metrics_stream(
      logger)
      .close();

  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(logger) = true;
  logger.log_request_rejected(6);

  std::ifstream stream(metrics_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.empty());

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, RequestRejectedWritesQueueMetricsWhenReady)
{
  BatchingTraceLogger logger;
  const auto trace_path = make_temp_trace_path();
  remove_trace_outputs(trace_path);

  auto metrics_path = trace_path;
  metrics_path.replace_filename("metrics.csv");

  starpu_server::testing::BatchingTraceLoggerTestAccessor::enabled_flag(logger)
      .store(true, std::memory_order_release);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::warmup_suppressed(
      logger)
      .store(false, std::memory_order_release);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      trace_start_initialized(logger) = true;
  starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_start_us(
      logger) = 0;
  starpu_server::testing::BatchingTraceLoggerTestAccessor::rejected_total(
      logger)
      .store(0, std::memory_order_relaxed);

  starpu_server::testing::BatchingTraceLoggerTestAccessor::queue_metrics_stream(
      logger)
      .open(metrics_path, std::ios::out | std::ios::trunc);
  ASSERT_TRUE(starpu_server::testing::BatchingTraceLoggerTestAccessor::
                  queue_metrics_stream(logger)
                      .is_open());
  logger.log_request_rejected(7);
  starpu_server::testing::BatchingTraceLoggerTestAccessor::queue_metrics_stream(
      logger)
      .close();

  std::ifstream stream(metrics_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_FALSE(content.empty());
  EXPECT_NE(content.find(",7,1"), std::string::npos);

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, QueueMetricWriteNoOpsWhenStreamClosed)
{
  BatchingTraceLogger logger;
  const auto trace_path = make_temp_trace_path();
  remove_trace_outputs(trace_path);

  auto metrics_path = trace_path;
  metrics_path.replace_filename("metrics.csv");

  const BatchingTraceLogger::QueueMetric metric{
      .timestamp_us = 123,
      .queue_size = 5,
      .rejected_total = 2,
  };
  starpu_server::testing::BatchingTraceLoggerTestAccessor::
      write_queue_metric_locked(logger, metric);

  EXPECT_FALSE(starpu_server::testing::BatchingTraceLoggerTestAccessor::
                   queue_metrics_stream(logger)
                       .is_open());
  EXPECT_FALSE(std::filesystem::exists(metrics_path));

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, TraceFileWriterHeaderNoOpsWhenStreamClosed)
{
  detail::TraceFileWriter writer;
  writer.write_header();

  EXPECT_FALSE(writer.is_open());
  EXPECT_FALSE(
      starpu_server::testing::TraceFileWriterTestAccessor::header_written(
          writer));
  EXPECT_FALSE(writer.ready());
  EXPECT_TRUE(
      starpu_server::testing::TraceFileWriterTestAccessor::thread_metadata(
          writer)
          .empty());
}

TEST(BatchingTraceLoggerTest, TraceFileWriterReportsOpenState)
{
  BatchingTraceLogger logger;
  const auto trace_path = make_temp_trace_path();

  logger.configure(true, trace_path.string());
  EXPECT_TRUE(
      starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
          logger)
          .is_open());

  logger.configure(false, "");
  EXPECT_FALSE(
      starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
          logger)
          .is_open());

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, ConfigureFromRuntimeUsesTraceSettings)
{
  BatchingTraceLogger logger;
  const auto trace_path = make_temp_trace_path();

  RuntimeConfig cfg;
  cfg.batching.trace_enabled = true;
  cfg.batching.trace_output_path = trace_path.string();

  logger.configure_from_runtime(cfg);

  EXPECT_TRUE(logger.enabled());
  EXPECT_EQ(
      starpu_server::testing::BatchingTraceLoggerTestAccessor::file_path(
          logger),
      trace_path.string());
  EXPECT_TRUE(std::filesystem::exists(trace_path));

  cfg.batching.trace_enabled = false;
  logger.configure_from_runtime(cfg);

  EXPECT_FALSE(logger.enabled());
  EXPECT_TRUE(
      starpu_server::testing::BatchingTraceLoggerTestAccessor::file_path(logger)
          .empty());

  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, ExposesSummaryFilePathWhenConfigured)
{
  BatchingTraceLogger logger;
  const auto trace_path = make_temp_trace_path();

  ASSERT_FALSE(logger.summary_file_path().has_value());
  logger.configure(true, trace_path.string());

  const auto summary_path = make_summary_path(trace_path);
  const auto summary = logger.summary_file_path();
  ASSERT_TRUE(summary.has_value());
  EXPECT_EQ(summary->string(), summary_path.string());

  logger.configure(false, "");
  remove_trace_outputs(trace_path);
}

TEST(BatchingTraceLoggerTest, SkipsWarmupSummaryEntries)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const std::array<int, 1> request_ids{1};
  const std::array<int64_t, 1> request_arrivals{500};
  logger.log_batch_summary(BatchingTraceLogger::BatchSummaryLogArgs{
      .batch_id = 7,
      .model_name = "warm_model",
      .batch_size = 1,
      .request_ids = request_ids,
      .request_arrival_us = request_arrivals,
      .worker_id = 2,
      .worker_type = DeviceType::CPU,
      .device_id = -1,
      .queue_ms = 0.1,
      .batch_ms = 0.2,
      .submit_ms = 0.3,
      .scheduling_ms = 0.4,
      .codelet_ms = 0.5,
      .inference_ms = 0.6,
      .callback_ms = 0.7,
      .total_ms = 0.8,
      .is_warmup = true,
  });
  logger.configure(false, "");

  const auto summary_path = make_summary_path(trace_path);
  const auto lines = read_summary_lines(summary_path);
  ASSERT_EQ(lines.size(), 1U);
  EXPECT_TRUE(lines[0].starts_with("batch_id,model_name,worker_id"))
      << "Unexpected header line for warmup-only export: " << lines[0];

  remove_trace_outputs(trace_path);
}

}}  // namespace starpu_server
