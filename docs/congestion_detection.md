# Congestion Detection System

## Overview

The StarPU Inference Server includes an automatic congestion detection system that monitors incoming request arrival rates and alerts when the system approaches its measured throughput capacity. This passive monitoring system helps identify performance bottlenecks and provides insights for capacity planning.

## Purpose

The congestion detection system serves three main purposes:

1. **Early Warning System** - Alerts operators when request load approaches system capacity
2. **Performance Analysis** - Correlates high load periods with latency spikes and throughput degradation
3. **Capacity Planning** - Identifies when scaling or optimization is needed

**Important:** This system is **passive monitoring only**. It does not throttle requests, reject connections, or actively manage load. It provides visibility into system stress for analysis and alerting.

## How It Works

### Basic Principle

The system compares the **arrival rate** of incoming requests against a **measured throughput** baseline established during server startup:

- **Measured Throughput**: Maximum sustainable processing rate (requests/second) determined through calibration
- **Arrival Rate**: Number of requests received in a sliding 1-second window
- **Congestion Threshold**: 95% of measured throughput (configurable)
- **Clear Threshold**: 90% of measured throughput (configurable)

### State Transitions

```
Normal State                    Congested State
(arrival < 95%)                (arrival >= 95%)
       │                               │
       │   arrival >= 95% threshold    │
       ├──────────────────────────────>│
       │                               │
       │   arrival < 90% threshold     │
       │<──────────────────────────────┤
       │                               │
```

**Congestion ENTERS when:**
- Arrival rate >= 95% of measured throughput
- Example: With 100 req/s capacity, congestion triggers at 95 req/s

**Congestion CLEARS when:**
- Arrival rate < 90% of measured throughput
- OR no requests received for 1 second (stale window)

The hysteresis (95% enter, 90% clear) prevents rapid state flapping.

## Implementation Architecture

### Components

1. **Throughput Measurement** ([server_main.cpp](../src/grpc/server/server_main.cpp))
   - Runs synthetic probe workload during startup
   - Measures actual system throughput under load
   - Caches result to `<config>_throughput.txt`

2. **Arrival Tracking** ([inference_service.cpp:465-542](../src/grpc/server/inference_service.cpp#L465-L542))
   - Records timestamp of each incoming request
   - Maintains sliding 1-second window of arrivals
   - Calculates current arrival rate

3. **Threshold Monitoring** ([inference_service.cpp:504-541](../src/grpc/server/inference_service.cpp#L504-L541))
   - Compares arrival rate against thresholds
   - Detects state transitions
   - Logs warnings and records events

4. **Background Monitor** ([inference_service.cpp:411-462](../src/grpc/server/inference_service.cpp#L411-L462))
   - Polls every 200ms
   - Detects stale windows (no activity)
   - Auto-clears congestion on staleness

### Data Structures

```cpp
// Configuration constants
constexpr double kCongestionEnterRatio = 0.95;      // 95% threshold
constexpr double kCongestionClearRatio = 0.90;      // 90% threshold
constexpr auto kArrivalWindow = std::chrono::seconds(1);
constexpr auto kCongestionMonitorPeriod = std::chrono::milliseconds(200);

// Runtime state
double measured_throughput_;                        // Baseline capacity (req/s)
double congestion_threshold_;                       // Enter threshold (95%)
double congestion_clear_threshold_;                 // Clear threshold (90%)
std::deque<time_point> recent_arrivals_;           // Sliding window
bool congestion_active_;                            // Current state
```

## Throughput Measurement

The `measured_throughput` baseline is established through a multi-phase probing process at server startup:

### Phase 1: Calibration

1. Generates synthetic workload: `calibration_multiplier * batches_per_worker` requests
2. Processes requests through inference pipeline
3. Measures actual throughput

### Phase 2: Duration-Calibrated Measurement

1. Adjusts request count to target 10-15 seconds runtime
2. Runs final measurement probe with calibrated load
3. Computes final throughput value: `requests / elapsed_time`

### Phase 3: Caching

- Result saved to `<config>_throughput.txt`
- Example: `batching_trace_throughput.txt`
- Reused on subsequent runs (skip probing)
- Delete cache file to force re-measurement

**Note:** Probing only runs when CUDA is enabled. Without CUDA, `measured_throughput = 0.0` and congestion detection is disabled.

## Configuration

### Automatic Enablement

Congestion detection is **automatically enabled** when:
- `measured_throughput > 0.0` (CUDA enabled and probe completed)

It is **automatically disabled** when:
- `measured_throughput <= 0.0` (CUDA disabled or probe skipped)

### Tuning Parameters

To adjust detection sensitivity, modify constants in [inference_service.cpp:55-58](../src/grpc/server/inference_service.cpp#L55-L58):

```cpp
constexpr double kCongestionEnterRatio = 0.95;  // Lower = more sensitive
constexpr double kCongestionClearRatio = 0.90;  // Higher = slower clear
constexpr auto kArrivalWindow = std::chrono::seconds(1);  // Averaging window
constexpr auto kCongestionMonitorPeriod = std::chrono::milliseconds(200);  // Poll rate
```

**Recommendations:**
- **kCongestionEnterRatio**: 0.90-0.95 (higher = fewer alerts)
- **kCongestionClearRatio**: 0.85-0.90 (lower than enter ratio for hysteresis)
- **kArrivalWindow**: 1-5 seconds (shorter = more responsive, noisier)
- **kCongestionMonitorPeriod**: 100-500ms (shorter = faster stale detection)

### Disabling Detection

To completely disable congestion detection:

1. Skip throughput probing by setting `measured_throughput = 0.0` in configuration
2. OR disable CUDA (forces throughput = 0.0)
3. OR delete the throughput cache file to prevent loading

## Monitoring and Observability

### Log Messages

Congestion state transitions are logged as warnings:

```
[Congestion] GPUs congestion detected: arrival rate 95.00 req/s is near measured throughput 100.00 infer/s
[Congestion] GPUs congestion cleared: arrival rate 85.00 req/s is below 90.00 infer/s
```

These messages include:
- Current arrival rate
- Measured throughput baseline
- Threshold values

### Trace Events

Congestion periods are recorded in `batching_trace.json` as trace events:

```json
{
  "name": "congestion",
  "ph": "X",
  "ts": 1234567890,
  "dur": 5000000,
  "args": {
    "enter_threshold": 95.0,
    "clear_threshold": 90.0,
    "measured_throughput": 100.0
  }
}
```

Fields:
- **ts**: Start timestamp (microseconds)
- **dur**: Duration (microseconds)
- **enter_threshold**: Congestion enter threshold (req/s)
- **clear_threshold**: Congestion clear threshold (req/s)
- **measured_throughput**: Baseline capacity (req/s)

### Visualization

The `plot_batch_summary.py` script automatically visualizes congestion periods:

```bash
python scripts/plot_batch_summary.py batching_trace.json
```

Congestion zones appear as:
- **Pink shaded regions** overlaid on latency/throughput plots
- **Semi-transparent** (alpha 0.35) to show underlying data
- **Aligned with batch IDs** for correlation analysis

This enables visual correlation between:
- High arrival rates (congestion periods)
- Latency spikes
- Batch processing delays
- Throughput degradation

## Use Cases

### 1. Performance Debugging

**Scenario:** Users report intermittent high latency

**Analysis:**
1. Review trace visualization with congestion overlay
2. Correlate latency spikes with congestion periods
3. Determine if load is the root cause
4. Assess if scaling is needed vs. optimization

### 2. Capacity Planning

**Scenario:** Planning for production deployment

**Analysis:**
1. Run load tests at various traffic levels
2. Monitor when congestion occurs
3. Measure performance degradation during congestion
4. Size infrastructure to stay below congestion threshold

### 3. Alerting and Monitoring

**Scenario:** Production monitoring setup

**Implementation:**
1. Parse server logs for congestion warnings
2. Set up alerts when congestion detected
3. Track frequency and duration of congestion
4. Correlate with business metrics (error rates, user complaints)

### 4. A/B Testing

**Scenario:** Testing model optimizations

**Analysis:**
1. Measure baseline throughput before changes
2. Deploy optimized model
3. Compare throughput increase (higher congestion threshold)
4. Quantify improvement in capacity

## Limitations

1. **Passive Only**: Does not throttle or reject requests
2. **No Backpressure**: Clients are not notified of congestion
3. **Simple Metric**: Only considers arrival rate, not queue depth or latency
4. **Static Threshold**: Does not adapt to changing system conditions
5. **Single Metric**: Does not account for request complexity variation

## Integration with Probe Mode

During throughput measurement, the system operates in special probe modes:

- **Calibration Mode**: Initial throughput estimation phase
- **Duration-Calibrated Mode**: Final measurement phase

Trace events during probing are prefixed:
- `prob_cal_*` for calibration
- `prob_dur_*` for duration-calibrated

A separate probe summary file is written: `batching_trace_probe_summary.csv`

This allows filtering out probe-related data from production analysis.

## Technical Details

### Thread Safety

The implementation uses mutex protection for shared state:

```cpp
std::mutex congestion_mutex_;  // Protects all congestion state
```

Accessed by:
- **Request handler threads**: Record arrivals
- **Monitor thread**: Check stale windows and auto-clear

### Performance Impact

- **Per-request overhead**: ~1 mutex lock/unlock + deque insertion
- **Background overhead**: 200ms polling with O(n) window cleanup
- **Memory overhead**: ~8 bytes per arrival in 1-second window

Typical memory usage: ~800 bytes for 100 req/s load

### Algorithm Complexity

- **Arrival recording**: O(1) amortized (deque push)
- **Rate calculation**: O(n) where n = arrivals in window (typically < 1000)
- **Threshold check**: O(1)

## Future Enhancements

Potential improvements for consideration:

1. **Dynamic Thresholds**: Adapt based on recent performance
2. **Request Complexity Weighting**: Account for varying request costs
3. **Queue Depth Integration**: Consider pending work, not just arrivals
4. **Client Feedback**: Return congestion signals in gRPC metadata
5. **Multi-level Thresholds**: Warning (80%), congestion (95%), critical (100%)
6. **Predictive Detection**: Forecast congestion before it occurs

## References

- Implementation: [inference_service.cpp](../src/grpc/server/inference_service.cpp)
- Configuration: [inference_service.hpp](../src/grpc/server/inference_service.hpp)
- Throughput Probing: [server_main.cpp](../src/grpc/server/server_main.cpp)
- Trace Logging: [batching_trace_logger.cpp](../src/utils/batching_trace_logger.cpp)
- Visualization: [plot_batch_summary.py](../scripts/plot_batch_summary.py)

## Summary

The congestion detection system provides lightweight, passive monitoring of system load by comparing incoming request rates against measured capacity. It integrates seamlessly with the existing tracing infrastructure to provide visibility into performance bottlenecks without impacting request processing.

Key benefits:
- **Zero configuration** - automatically enabled with CUDA
- **Low overhead** - simple sliding window algorithm
- **Rich telemetry** - logs, traces, and visualizations
- **Actionable insights** - identifies when scaling is needed
