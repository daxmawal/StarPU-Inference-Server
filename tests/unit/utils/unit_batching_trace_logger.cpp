#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
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

TEST(BatchingTraceLoggerTest, RoutesInvalidWorkerEventsToQueueTrack)
{
  const auto trace_path = make_temp_trace_path();
  auto& logger = BatchingTraceLogger::instance();

  logger.configure(true, trace_path.string());
  logger.log_request_enqueued(1, "demo_model");
  logger.log_batch_submitted(7, "demo_model", 1, 1, 0, DeviceType::CPU);
  logger.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());

  EXPECT_NE(content.find("\"tid\":1"), std::string::npos);
  EXPECT_NE(content.find("\"worker_id\":-1"), std::string::npos);
  EXPECT_NE(content.find("task_queue"), std::string::npos);
  EXPECT_NE(content.find("\"worker_id\":0"), std::string::npos);
  EXPECT_NE(content.find("worker-0 (cpu)"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(trace_path, ec);
}

}}  // namespace starpu_server
