#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "monitoring/metrics.hpp"

using namespace starpu_server;

TEST(Metrics, ConcurrentInit)
{
  const int thread_count = 8;
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (int i = 0; i < thread_count; ++i) {
    threads.emplace_back([]() { init_metrics(0); });
  }
  for (auto& t : threads) {
    t.join();
  }
  EXPECT_NE(metrics.load(std::memory_order_acquire), nullptr);
  shutdown_metrics();
}

TEST(Metrics, ConcurrentShutdown)
{
  init_metrics(0);
  EXPECT_NE(metrics.load(std::memory_order_acquire), nullptr);

  const int thread_count = 8;
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (int i = 0; i < thread_count; ++i) {
    threads.emplace_back([]() { shutdown_metrics(); });
  }
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(metrics.load(std::memory_order_acquire), nullptr);
}
