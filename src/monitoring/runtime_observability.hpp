#pragma once

#include <memory>

namespace starpu_server {

class BatchingTraceLogger;
class MetricsRecorder;

namespace congestion {
class Monitor;
}

struct RuntimeObservability {
  std::shared_ptr<MetricsRecorder> metrics;
  std::shared_ptr<BatchingTraceLogger> tracer;
  std::shared_ptr<congestion::Monitor> congestion_monitor;
};

}  // namespace starpu_server
