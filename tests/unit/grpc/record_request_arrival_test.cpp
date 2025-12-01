#include <gtest/gtest.h>
#include <torch/torch.h>

#include <chrono>
#include <memory>
#include <vector>

#define private public
#include "grpc/server/inference_service.hpp"
#undef private

namespace {

class RecordRequestArrivalTest : public ::testing::Test {
 protected:
  RecordRequestArrivalTest() = default;
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

TEST_F(RecordRequestArrivalTest, EarlyReturnWhenMeasuredThroughputIsZero)
{
  auto service = make_service(0.0);
  const auto now = std::chrono::high_resolution_clock::now();
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_EQ(service->recent_arrivals_.size(), 0);
  }
  service->record_request_arrival(now);
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_EQ(service->recent_arrivals_.size(), 0);
  }
}

TEST_F(RecordRequestArrivalTest, EarlyReturnWhenMeasuredThroughputIsNegative)
{
  auto service = make_service(-10.0);
  const auto now = std::chrono::high_resolution_clock::now();
  service->record_request_arrival(now);
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_EQ(service->recent_arrivals_.size(), 0);
  }
}

TEST_F(RecordRequestArrivalTest, SingleArrivalIsRecorded)
{
  auto service = make_service(100.0, true, 0.0);
  const auto now = std::chrono::high_resolution_clock::now();
  service->record_request_arrival(now);
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_EQ(service->recent_arrivals_.size(), 1);
    EXPECT_EQ(service->recent_arrivals_.front(), now);
    EXPECT_EQ(service->last_arrival_time_, now);
  }
}

TEST_F(RecordRequestArrivalTest, MultipleArrivalsWithinWindowAreKept)
{
  auto service = make_service(100.0, true, 0.0);
  const auto now = std::chrono::high_resolution_clock::now();
  service->record_request_arrival(now - std::chrono::milliseconds(500));
  service->record_request_arrival(now - std::chrono::milliseconds(300));
  service->record_request_arrival(now);
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_EQ(service->recent_arrivals_.size(), 3);
    EXPECT_EQ(service->last_arrival_time_, now);
  }
}

TEST_F(RecordRequestArrivalTest, OldArrivalsOutsideWindowAreRemoved)
{
  auto service = make_service(100.0, true, 0.0);
  const auto now = std::chrono::high_resolution_clock::now();
  const auto old_time = now - std::chrono::seconds(2);
  const auto recent_time = now - std::chrono::milliseconds(500);
  service->record_request_arrival(old_time);
  service->record_request_arrival(recent_time);
  service->record_request_arrival(now);
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_EQ(service->recent_arrivals_.size(), 2);
    EXPECT_NE(service->recent_arrivals_.front(), old_time);
  }
}

TEST_F(
    RecordRequestArrivalTest, CongestionDetectedWhenArrivalRateExceedsThreshold)
{
  auto service = make_service(100.0, true, 0.0);
  const auto now = std::chrono::high_resolution_clock::now();
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_FALSE(service->congestion_active_);
  }
  for (int i = 0; i < 96; ++i) {
    service->record_request_arrival(now + std::chrono::milliseconds(i * 10));
  }
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_TRUE(service->congestion_active_);
    EXPECT_NE(
        service->congestion_start_time_,
        std::chrono::high_resolution_clock::time_point{});
  }
}

TEST_F(
    RecordRequestArrivalTest,
    CongestionClearedWhenArrivalRateFallsBelowThreshold)
{
  auto service = make_service(100.0, true, 0.0);
  const auto now = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 96; ++i) {
    service->record_request_arrival(now + std::chrono::milliseconds(i * 10));
  }
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_TRUE(service->congestion_active_);
  }
  const auto later_time = now + std::chrono::seconds(2);
  for (int i = 0; i < 89; ++i) {
    service->record_request_arrival(
        later_time + std::chrono::milliseconds(i * 11));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_FALSE(service->congestion_active_);
  }
}

TEST_F(RecordRequestArrivalTest, NoStateChangeWhenArrivalRateBetweenThresholds)
{
  auto service = make_service(100.0, true, 0.0);
  const auto now = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 92; ++i) {
    service->record_request_arrival(now + std::chrono::milliseconds(i * 10));
  }
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_FALSE(service->congestion_active_);
  }
}

TEST_F(RecordRequestArrivalTest, LastArrivalTimeUpdatedOnEachCall)
{
  auto service = make_service(100.0, true, 0.0);
  const auto time1 = std::chrono::high_resolution_clock::now();
  const auto time2 = time1 + std::chrono::milliseconds(100);
  const auto time3 = time2 + std::chrono::milliseconds(100);
  service->record_request_arrival(time1);
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_EQ(service->last_arrival_time_, time1);
  }
  service->record_request_arrival(time2);
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_EQ(service->last_arrival_time_, time2);
  }
  service->record_request_arrival(time3);
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_EQ(service->last_arrival_time_, time3);
  }
}

TEST_F(RecordRequestArrivalTest, ArrivalRateCalculationIsCorrect)
{
  auto service = make_service(100.0, true, 0.0);
  const auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; ++i) {
    service->record_request_arrival(
        start_time + std::chrono::milliseconds(i * 100));
  }
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_EQ(service->recent_arrivals_.size(), 10);
  }
}

TEST_F(RecordRequestArrivalTest, GPUDeviceNameUsedWhenUseCudaTrue)
{
  auto service = make_service(100.0, true, 0.0);
  EXPECT_TRUE(service->use_cuda_);
}

TEST_F(RecordRequestArrivalTest, CPUDeviceNameUsedWhenUseCudaFalse)
{
  auto service = make_service(0.0, false, 100.0);
  EXPECT_FALSE(service->use_cuda_);
}

TEST_F(RecordRequestArrivalTest, ThresholdsAreCapturedAtRecordTime)
{
  auto service = make_service(100.0, true, 0.0);
  const auto now = std::chrono::high_resolution_clock::now();
  EXPECT_DOUBLE_EQ(service->congestion_threshold_, 95.0);
  EXPECT_DOUBLE_EQ(service->congestion_clear_threshold_, 90.0);
  service->record_request_arrival(now);
  service->record_request_arrival(now + std::chrono::milliseconds(100));
}

TEST_F(RecordRequestArrivalTest, EmptyArrivalsAfterFilteringCausesEarlyReturn)
{
  auto service = make_service(100.0, true, 0.0);
  const auto now = std::chrono::high_resolution_clock::now();
  {
    std::scoped_lock lock(service->congestion_mutex_);
    service->recent_arrivals_.push_back(now - std::chrono::seconds(5));
  }
  service->record_request_arrival(now);
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_EQ(service->recent_arrivals_.size(), 1);
  }
}

TEST_F(RecordRequestArrivalTest, CongestionStartTimeSetWhenEntering)
{
  auto service = make_service(100.0, true, 0.0);
  const auto now = std::chrono::high_resolution_clock::now();
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_EQ(
        service->congestion_start_time_,
        std::chrono::high_resolution_clock::time_point{});
  }
  for (int i = 0; i < 96; ++i) {
    service->record_request_arrival(now + std::chrono::milliseconds(i * 10));
  }
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_NE(
        service->congestion_start_time_,
        std::chrono::high_resolution_clock::time_point{});
  }
}

TEST_F(RecordRequestArrivalTest, CongestionStartTimeClaredWhenExitingCongestion)
{
  auto service = make_service(100.0, true, 0.0);
  const auto now = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 96; ++i) {
    service->record_request_arrival(now + std::chrono::milliseconds(i * 10));
  }
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_TRUE(service->congestion_active_);
    EXPECT_NE(
        service->congestion_start_time_,
        std::chrono::high_resolution_clock::time_point{});
  }
  const auto later_time = now + std::chrono::seconds(2);
  for (int i = 0; i < 89; ++i) {
    service->record_request_arrival(
        later_time + std::chrono::milliseconds(i * 11));
  }
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_FALSE(service->congestion_active_);
    EXPECT_EQ(
        service->congestion_start_time_,
        std::chrono::high_resolution_clock::time_point{});
  }
}

TEST_F(RecordRequestArrivalTest, MutexProtectsSharedState)
{
  auto service = make_service(100.0, true, 0.0);
  const auto now = std::chrono::high_resolution_clock::now();
  service->record_request_arrival(now);
  service->record_request_arrival(now + std::chrono::milliseconds(10));
  service->record_request_arrival(now + std::chrono::milliseconds(20));
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_EQ(service->recent_arrivals_.size(), 3);
  }
}

TEST_F(RecordRequestArrivalTest, DequeMainainsChronologicalOrder)
{
  auto service = make_service(100.0, true, 0.0);
  const auto now = std::chrono::high_resolution_clock::now();

  const auto time1 = now;
  const auto time2 = now + std::chrono::milliseconds(100);
  const auto time3 = now + std::chrono::milliseconds(200);

  service->record_request_arrival(time1);
  service->record_request_arrival(time2);
  service->record_request_arrival(time3);

  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_EQ(service->recent_arrivals_[0], time1);
    EXPECT_EQ(service->recent_arrivals_[1], time2);
    EXPECT_EQ(service->recent_arrivals_[2], time3);
  }
}

TEST_F(RecordRequestArrivalTest, NoCongestionEnterIfOnlyAtEnterThreshold)
{
  auto service = make_service(100.0, true, 0.0);
  const auto now = std::chrono::high_resolution_clock::now();
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_FALSE(service->congestion_active_);
  }
  for (int i = 0; i < 95; ++i) {
    service->record_request_arrival(now + std::chrono::milliseconds(i * 10));
  }
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_TRUE(service->congestion_active_);
  }
}

TEST_F(RecordRequestArrivalTest, CongestionRemainsClearWhenArrivalsDropLow)
{
  auto service = make_service(100.0, true, 0.0);
  const auto now = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 96; ++i) {
    service->record_request_arrival(now + std::chrono::milliseconds(i * 10));
  }
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_TRUE(service->congestion_active_);
  }
  {
    std::scoped_lock lock(service->congestion_mutex_);
    EXPECT_NE(
        service->congestion_start_time_,
        std::chrono::high_resolution_clock::time_point{});
  }
}

}  // namespace
