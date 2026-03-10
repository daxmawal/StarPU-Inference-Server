#pragma once

#include <prometheus/histogram.h>

#include <chrono>
#include <cstddef>
#include <filesystem>

namespace starpu_server::inline metrics_internal_detail {

const std::filesystem::path kProcStatm{"/proc/self/statm"};

const prometheus::Histogram::BucketBoundaries kInferenceLatencyMsBuckets{
    1, 5, 10, 25, 50, 100, 250, 500, 1000};
const prometheus::Histogram::BucketBoundaries kBatchSizeBuckets{
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
const prometheus::Histogram::BucketBoundaries kBatchEfficiencyBuckets{
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0, 8.0};
const prometheus::Histogram::BucketBoundaries kModelLoadDurationMsBuckets{
    10, 50, 100, 200, 500, 1000, 2000, 5000, 10000};
const prometheus::Histogram::BucketBoundaries kTaskRuntimeMsBuckets{
    1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000};

constexpr std::size_t kMaxLabelSeries = 10000;

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
constexpr auto kSamplingErrorLogThrottle = std::chrono::seconds(0);
#else
constexpr auto kSamplingErrorLogThrottle = std::chrono::seconds(60);
#endif  // SONAR_IGNORE_END

}  // namespace starpu_server::inline metrics_internal_detail
