# Congestion Detection System (Summary)

## Overview

The congestion detection system monitors incoming request rates and warns when the server approaches its maximum measured throughput.  
It is **passive only**, it does not throttle, reject, or regulate load.

## How It Works

The system compares:

- **Arrival rate**: number of requests in a 1-second sliding window  
- **Measured throughput**: calibrated capacity obtained at startup  
- **Thresholds**:  
  - **95%** → enter congestion  
  - **90%** → clear congestion  

This hysteresis prevents rapid state toggling.

## State Diagram

```text
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

## State Logic

- **Enter** when: `arrival_rate ≥ 0.95 * measured_throughput`  
- **Clear** when: `arrival_rate < 0.90 * measured_throughput` or no arrivals for 1 second.

A background thread (200 ms interval) cleans old timestamps and checks transitions.

## Throughput Measurement

At startup, the server runs a synthetic probe to estimate:

```
measured_throughput = total_requests / elapsed_time
```

The value is cached in `<config>_throughput.txt`.

## Key Configuration

```cpp
constexpr double kCongestionEnterRatio = 0.95;
constexpr double kCongestionClearRatio = 0.90;
constexpr auto kArrivalWindow = std::chrono::seconds(1);
constexpr auto kCongestionMonitorPeriod = std::chrono::milliseconds(200);
```

## Logs & Traces

```
[Congestion] GPUs congestion detected: ...
[Congestion] GPUs congestion cleared: ...
```

Congestion periods appear in `batching_trace.json` and can be visualized (pink regions).
