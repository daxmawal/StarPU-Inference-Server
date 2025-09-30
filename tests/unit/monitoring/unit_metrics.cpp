#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <string_view>
#include <vector>

#include "monitoring/metrics.hpp"

using namespace starpu_server;

namespace {
void
AssertMetricsInitialized(const std::shared_ptr<MetricsRegistry>& metrics)
{
  ASSERT_NE(metrics, nullptr);
  ASSERT_NE(metrics->registry, nullptr);
  ASSERT_NE(metrics->requests_total, nullptr);
  ASSERT_NE(metrics->inference_latency, nullptr);
  ASSERT_NE(metrics->queue_size_gauge, nullptr);
}

auto
HasMetric(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view name) -> bool
{
  return std::ranges::any_of(
      families, [name](const prometheus::MetricFamily& family) {
        return family.name == name;
      });
}
}  // namespace

TEST(Metrics, InitializesPointersAndRegistry)
{
  ASSERT_TRUE(init_metrics(0));

  auto metrics = get_metrics();
  AssertMetricsInitialized(metrics);

  const auto families = metrics->registry->Collect();
  EXPECT_TRUE(HasMetric(families, "requests_total"));
  EXPECT_TRUE(HasMetric(families, "inference_latency_ms"));
  EXPECT_TRUE(HasMetric(families, "inference_queue_size"));

  shutdown_metrics();
  EXPECT_EQ(get_metrics(), nullptr);
}

TEST(Metrics, RepeatedInitDoesNotAllocateRegistry)
{
  ASSERT_TRUE(init_metrics(0));
  auto first = get_metrics();

  EXPECT_FALSE(init_metrics(0));
  auto second = get_metrics();
  EXPECT_EQ(first, second);

  shutdown_metrics();
  EXPECT_EQ(get_metrics(), nullptr);
}

TEST(Metrics, InitFailsWhenMetricsRegistryThrows)
{
  shutdown_metrics();

  int reserved_socket = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(reserved_socket, 0);

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;

  ASSERT_EQ(
      ::bind(reserved_socket, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)),
      0);
  ASSERT_EQ(::listen(reserved_socket, 1), 0);

  socklen_t addr_len = sizeof(addr);
  ASSERT_EQ(
      ::getsockname(
          reserved_socket, reinterpret_cast<sockaddr*>(&addr), &addr_len),
      0);
  const int reserved_port = ntohs(addr.sin_port);
  ASSERT_GT(reserved_port, 0);

  EXPECT_FALSE(init_metrics(reserved_port));
  EXPECT_EQ(get_metrics(), nullptr);

  ::close(reserved_socket);


  shutdown_metrics();
  EXPECT_EQ(get_metrics(), nullptr);
}
