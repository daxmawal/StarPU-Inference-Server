#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <span>
#include <string>

#include "utils/batching_trace_logger.hpp"

namespace starpu_server { namespace {

auto
make_temp_trace_path() -> std::filesystem::path
{
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         std::format("batching_trace_test_{}.json", now);
}

TEST(BatchingTraceLoggerTest, RoutesNonWorkerEventsToDedicatedTracks)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  logger.log_request_enqueued(1, "demo_model");
  const auto enqueue_start = std::chrono::high_resolution_clock::now();
  const auto enqueue_end = enqueue_start + std::chrono::microseconds(10);
  const std::array<int, 2> request_ids{0, 1};
  logger.log_batch_enqueue_span(
      5, "demo_model", request_ids.size(), enqueue_start, enqueue_end,
      std::span<const int>(request_ids));
  logger.log_batch_submitted(5, "demo_model", 1);
  logger.log_batch_submitted(7, "demo_model", 1, 0, DeviceType::CPU);
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

  std::error_code ec;
  std::filesystem::remove(trace_path, ec);
}

TEST(BatchingTraceLoggerTest, IncludesDeviceIdInWorkerLabels)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const auto start = std::chrono::high_resolution_clock::now();
  const auto end = start + std::chrono::microseconds(50);
  logger.log_batch_submitted(
      3, "demo_model", 1, 4, DeviceType::CUDA, std::span<const int>{},
      /*is_warmup=*/false, /*device_id=*/7);
  logger.log_batch_compute_span(
      3, "demo_model", 1, 4, DeviceType::CUDA, start, end,
      /*is_warmup=*/false, /*device_id=*/7);
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());

  EXPECT_NE(content.find("device 7 worker 4 (cuda)"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(trace_path, ec);
}

TEST(BatchingTraceLoggerTest, RecordsRequestIdsForBatchSubmission)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const std::array<int, 3> request_ids{42, 43, 44};
  logger.log_batch_submitted(
      9, "demo_model", 2, 1, DeviceType::CUDA,
      std::span<const int>(request_ids));
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

  std::error_code ec;
  std::filesystem::remove(trace_path, ec);
}

TEST(BatchingTraceLoggerTest, EmitsBatchBuildSpanWithRequestIds)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const auto start = std::chrono::high_resolution_clock::now();
  const auto end = start + std::chrono::microseconds(150);
  const std::array<int, 2> request_ids{7, 8};
  logger.log_batch_build_span(
      21, "demo_model", 9, start, end, std::span<const int>(request_ids));
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

  std::error_code ec;
  std::filesystem::remove(trace_path, ec);
}

TEST(BatchingTraceLoggerTest, EmitsBatchEnqueueSpanWithRequestIds)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const auto start = std::chrono::high_resolution_clock::now();
  const auto end = start + std::chrono::microseconds(60);
  const std::array<int, 3> request_ids{4, 5, 6};
  logger.log_batch_enqueue_span(
      21, "demo_model", request_ids.size(), start, end,
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

  std::error_code ec;
  std::filesystem::remove(trace_path, ec);
}

TEST(BatchingTraceLoggerTest, ExtendsEnqueueWindowToLatestRequestEvent)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const auto base = std::chrono::high_resolution_clock::now();
  const auto first = base + std::chrono::microseconds(10);
  const auto second = first + std::chrono::microseconds(5);
  const auto last = first + std::chrono::microseconds(25);
  logger.log_request_enqueued(101, "demo_model", false, first);
  logger.log_request_enqueued(102, "demo_model", false, second);
  logger.log_request_enqueued(103, "demo_model", false, last);
  const std::array<int, 3> request_ids{101, 102, 103};
  logger.log_batch_enqueue_span(
      99, "demo_model", request_ids.size(), first, second,
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

  std::error_code ec;
  std::filesystem::remove(trace_path, ec);
}

TEST(BatchingTraceLoggerTest, PrefixesWarmupEvents)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const auto start = std::chrono::high_resolution_clock::now();
  const auto end = start + std::chrono::microseconds(100);
  const std::array<int, 2> request_ids{1, 2};
  logger.log_batch_submitted(
      11, "demo_model", 1, 0, DeviceType::CPU,
      std::span<const int>(request_ids), /*is_warmup=*/true);
  logger.log_batch_build_span(
      11, "demo_model", 1, start, end, std::span<const int>(request_ids),
      /*is_warmup=*/true);
  logger.log_batch_compute_span(
      11, "demo_model", 1, 0, DeviceType::CPU, start, end,
      /*is_warmup=*/true);
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
  EXPECT_NE(content.find("\"name\":\"warming_inference\""), std::string::npos);
  EXPECT_NE(
      content.find(
          "\"warming_batch_size\":1,\"warming_model_name\":\"demo_model\","
          "\"warming_worker_id\""),
      std::string::npos);
  EXPECT_EQ(content.find("\"warming_sample_count\""), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(trace_path, ec);
}

TEST(BatchingTraceLoggerTest, SplitsOverlappingComputeSpansIntoWorkerLanes)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const auto base = std::chrono::high_resolution_clock::now();
  const auto overlapping_start = base + std::chrono::microseconds(50);
  const auto overlapping_end =
      overlapping_start + std::chrono::microseconds(40);
  const auto long_end = base + std::chrono::microseconds(180);

  logger.log_batch_compute_span(
      1, "demo_model", 1, 0, DeviceType::CPU, overlapping_start,
      overlapping_end);
  logger.log_batch_compute_span(
      2, "demo_model", 1, 0, DeviceType::CPU, base, long_end);
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

  const auto first_compute = content.find("\"name\":\"inference\"");
  ASSERT_NE(first_compute, std::string::npos);
  const auto second_compute =
      content.find("\"name\":\"inference\"", first_compute + 1);
  ASSERT_NE(second_compute, std::string::npos);

  const int tid_one = extract_tid(first_compute);
  const int tid_two = extract_tid(second_compute);
  EXPECT_NE(tid_one, tid_two);
  EXPECT_NE(content.find("worker-0 (cpu) #2"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(trace_path, ec);
}

TEST(BatchingTraceLoggerTest, EmitsScopedFlowsBetweenSubmissionAndCompute)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  const auto start = std::chrono::high_resolution_clock::now();
  const auto end = start + std::chrono::microseconds(75);
  const auto build_start = start - std::chrono::microseconds(50);
  const auto build_end = build_start + std::chrono::microseconds(20);
  const auto warm_build_start = end + std::chrono::microseconds(50);
  const auto warm_build_end = warm_build_start + std::chrono::microseconds(20);
  const auto warm_start = warm_build_end + std::chrono::microseconds(25);
  const auto warm_end = warm_start + std::chrono::microseconds(75);
  logger.log_batch_build_span(
      0, "demo_model", 2, build_start, build_end, std::span<const int>{});
  logger.log_batch_submitted(0, "demo_model", 2, 0, DeviceType::CPU);
  logger.log_batch_compute_span(
      0, "demo_model", 2, 0, DeviceType::CPU, start, end);
  logger.log_batch_build_span(
      0, "demo_model", 1, warm_build_start, warm_build_end,
      std::span<const int>{}, /*is_warmup=*/true);
  logger.log_batch_submitted(
      0, "demo_model", 1, 1, DeviceType::CPU, std::span<const int>{},
      /*is_warmup=*/true);
  logger.log_batch_compute_span(
      0, "demo_model", 1, 1, DeviceType::CPU, warm_start, warm_end,
      /*is_warmup=*/true);
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  const auto serving_compute = content.find("\"name\":\"inference\"");
  ASSERT_NE(serving_compute, std::string::npos);
  const auto warming_compute = content.find("\"name\":\"warming_inference\"");
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

  std::error_code ec;
  std::filesystem::remove(trace_path, ec);
}

}}  // namespace starpu_server
