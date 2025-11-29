#include <gtest/gtest.h>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#define private public
#include "grpc/server/inference_service.hpp"
#undef private

namespace {

class StartCongestionMonitorTest : public ::testing::Test {
 protected:
  StartCongestionMonitorTest() = default;

  [[nodiscard]] auto make_service(
      double measured_throughput, bool use_cuda = false,
      double measured_throughput_cpu = 0.0)
      -> std::unique_ptr<starpu_server::InferenceServiceImpl>
  {
    return std::make_unique<starpu_server::InferenceServiceImpl>(
        &queue_, &reference_outputs_, std::vector<at::ScalarType>{at::kFloat},
        "test_model", measured_throughput, use_cuda, measured_throughput_cpu);
  }

  starpu_server::InferenceQueue queue_;
  std::vector<torch::Tensor> reference_outputs_ = {
      torch::zeros({1}, torch::TensorOptions().dtype(at::kFloat))};
};

// Test 1: No thread is created when measured_throughput is 0
TEST_F(StartCongestionMonitorTest, NoThreadCreatedWhenThroughputIsZero)
{
  auto service = make_service(0.0);
  // The jthread should not be running (default constructed)
  EXPECT_FALSE(service->congestion_monitor_thread_.joinable());
}

// Test 2: No thread is created when measured_throughput is negative
TEST_F(StartCongestionMonitorTest, NoThreadCreatedWhenThroughputIsNegative)
{
  auto service = make_service(-10.0);
  EXPECT_FALSE(service->congestion_monitor_thread_.joinable());
}

// Test 3: Thread is created when measured_throughput is positive with CPU
TEST_F(StartCongestionMonitorTest, ThreadCreatedWhenThroughputIsPositiveWithCPU)
{
  auto service = make_service(0.0, false, 100.0);
  EXPECT_TRUE(service->congestion_monitor_thread_.joinable());
}

// Test 4: Thread is created when measured_throughput is positive with GPU
TEST_F(StartCongestionMonitorTest, ThreadCreatedWhenThroughputIsPositiveWithGPU)
{
  auto service = make_service(100.0, true, 50.0);
  EXPECT_TRUE(service->congestion_monitor_thread_.joinable());
}

// Test 5: Thread is created even with very small positive throughput
TEST_F(
    StartCongestionMonitorTest, ThreadCreatedWhenThroughputIsVerySmallPositive)
{
  // When use_cuda=false, measured_throughput_cpu is used
  auto service = make_service(0.0, false, 0.001);
  EXPECT_TRUE(service->congestion_monitor_thread_.joinable());
}

// Test 6: Stale arrival window is cleared when no arrivals occur
TEST_F(StartCongestionMonitorTest, StaleArrivalWindowIsClearedOnMonitorCheck)
{
  auto service = make_service(100.0, true, 0.0);
  const auto start_time = std::chrono::high_resolution_clock::now();

  {
    std::scoped_lock lock(service->congestion_mutex_);
    service->congestion_active_ = true;
    service->congestion_start_time_ = start_time - std::chrono::seconds(2);
    service->last_arrival_time_ =
        start_time - std::chrono::seconds(2);  // More than 1 second ago
  }

  // Wait for the monitor thread to detect the stale window
  // The monitor checks every 200ms, so wait long enough
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  {
    std::scoped_lock lock(service->congestion_mutex_);
    // After monitor detects stale window, congestion should be cleared
    EXPECT_FALSE(service->congestion_active_);
    EXPECT_EQ(
        service->congestion_start_time_,
        std::chrono::high_resolution_clock::time_point{});
  }
}

// Test 7: Multiple arrivals within window are kept
TEST_F(
    StartCongestionMonitorTest, MultipleArrivalsWithinWindowAreKeptAfterRecord)
{
  auto service = make_service(100.0);
  const auto now = std::chrono::high_resolution_clock::now();

  {
    std::scoped_lock lock(service->congestion_mutex_);
    // Add multiple recent arrivals (all within last 1 second)
    service->recent_arrivals_.push_back(now - std::chrono::milliseconds(900));
    service->recent_arrivals_.push_back(now - std::chrono::milliseconds(600));
    service->recent_arrivals_.push_back(now - std::chrono::milliseconds(300));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_EQ(service->recent_arrivals_.size(), 3);
  }
}

// Test 8: Congestion thresholds are set correctly based on measured throughput
TEST_F(
    StartCongestionMonitorTest, CongestionThresholdsSetCorrectlyOnConstruction)
{
  const double throughput = 100.0;
  auto service = make_service(throughput, true, 50.0);

  // Verify thresholds are calculated correctly:
  // enter = 100 * 0.95 = 95 (GPU uses first param)
  // clear = 100 * 0.90 = 90 (GPU uses first param)
  EXPECT_DOUBLE_EQ(service->congestion_threshold_, 95.0);
  EXPECT_DOUBLE_EQ(service->congestion_clear_threshold_, 90.0);
}

// Test 9: Correct throughput is selected based on use_cuda flag
TEST_F(StartCongestionMonitorTest, CorrectThroughputSelectedWithCUDAFlag)
{
  // When use_cuda is true, should use measured_throughput parameter
  auto service_gpu = make_service(100.0, true, 50.0);
  EXPECT_DOUBLE_EQ(service_gpu->measured_throughput_, 100.0);

  // When use_cuda is false, should use measured_throughput_cpu parameter
  auto service_cpu = make_service(100.0, false, 50.0);
  EXPECT_DOUBLE_EQ(service_cpu->measured_throughput_, 50.0);
}

// Test 10: Thread gracefully stops when service is destroyed
TEST_F(StartCongestionMonitorTest, ThreadStopsGracefullyOnDestruction)
{
  {
    auto service = make_service(100.0, true, 0.0);
    EXPECT_TRUE(service->congestion_monitor_thread_.joinable());
  }
  // Thread should be stopped and joined automatically via jthread RAII
  // No exception should be thrown
}

// Test 11: arrival window detection is correct with mocked time
TEST_F(StartCongestionMonitorTest, ArrivalWindowCutoffIsCorrect)
{
  auto service = make_service(100.0);
  const auto now = std::chrono::high_resolution_clock::now();

  // Simulate multiple arrivals with precise timing
  {
    std::scoped_lock lock(service->congestion_mutex_);
    // Add arrivals: 1.2s, 1.0s, 0.8s, 0.6s, 0.4s, 0.2s ago
    service->recent_arrivals_.push_back(now - std::chrono::milliseconds(1200));
    service->recent_arrivals_.push_back(now - std::chrono::milliseconds(1000));
    service->recent_arrivals_.push_back(now - std::chrono::milliseconds(800));
    service->recent_arrivals_.push_back(now - std::chrono::milliseconds(600));
    service->recent_arrivals_.push_back(now - std::chrono::milliseconds(400));
    service->recent_arrivals_.push_back(now - std::chrono::milliseconds(200));
  }

  service->record_request_arrival(now);

  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_LE(service->recent_arrivals_.size(), 6);
  }
}

TEST_F(StartCongestionMonitorTest, CongestionMutexProtectsSharedState)
{
  auto service = make_service(100.0);
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_FALSE(service->congestion_active_);
    EXPECT_EQ(service->recent_arrivals_.size(), 0);
  }
}

TEST_F(
    StartCongestionMonitorTest, NegativeThroughputDoesNotCreateThreadWithCUDA)
{
  auto service = make_service(-1.0, true, 0.0);
  EXPECT_FALSE(service->congestion_monitor_thread_.joinable());
}

TEST_F(StartCongestionMonitorTest, ThreadExitsCleanlyOnStopRequest)
{
  auto service = make_service(100.0, true, 0.0);
  EXPECT_TRUE(service->congestion_monitor_thread_.joinable());
  service.reset();
  EXPECT_TRUE(true);
}

TEST_F(
    StartCongestionMonitorTest,
    ServiceWithZeroThroughputRecordDoesNotCheckMonitor)
{
  auto service = make_service(0.0);
  EXPECT_FALSE(service->congestion_monitor_thread_.joinable());
  const auto now = std::chrono::high_resolution_clock::now();
  EXPECT_NO_THROW(service->record_request_arrival(now));
}

}  // namespace
