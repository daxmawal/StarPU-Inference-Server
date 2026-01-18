# Congestion detection

This document explains the congestion detector, its signals, and the
configuration fields in the `congestion:` YAML block.

## Summary

- A background monitor samples queue and latency metrics every tick (default
  1000 ms, configurable via `tick_interval_ms`).
- Metrics are smoothed with an EWMA (Exponential Weighted Moving Average) to reduce noise.
- Congestion enters only after the entry horizon is satisfied.
- Congestion exits only after the exit horizon is satisfied.
- Any rejection in a tick triggers an immediate congestion state.

## Signals and derived metrics

- Arrival rate (lambda): requests/s over the last tick (lambda = arrivals / dt).
- Completion rate (mu): logical jobs/s over the last tick (mu = completions / dt).
- Load (rho): lambda / mu (smoothed; if mu == 0 and lambda > 0, rho is capped
  at 1000).
- Queue fill ratio: queue_size / queue_capacity (clamped to [0,1], smoothed).
- Queue growth rate: (queue_size - last_queue_size) / dt (smoothed).
- Queue latency p95: p95 of queue latency samples in the tick (smoothed).
- E2E latency p95: p95 of end-to-end latency samples in the tick (smoothed).

Notes:

- dt is the elapsed time since the previous tick, clamped to at least
  `tick_interval_ms`.
- Smoothed values use an EWMA: s_t = alpha *x_t + (1 - alpha)* s_{t-1}. The
  first sample initializes the EWMA, for percentile signals, if no samples
  arrive in a tick the smoothed value is unchanged.
- queue_budget_ms is `queue_latency_budget_ms` when set, otherwise
  `latency_slo_ms * queue_latency_budget_ratio` when SLO is enabled.

## Entry and exit logic (per tick)

Entry condition is true if ANY of the following holds:

- under_provisioned: rho_smoothed > rho_high
- queue_pressure: fill_smoothed > fill_high AND dqueue_smoothed > 0
- latency_danger:
  - if latency_slo_ms > 0: e2e_p95_smoothed > latency_slo_ms * e2e_warn_ratio
  - else if queue budget is set: queue_p95_smoothed > queue_budget_ms

Exit condition is true if ALL of the following holds:

- fill_smoothed < fill_low
- rho_smoothed < rho_low
- latency_ok:
  - if latency_slo_ms > 0: e2e_p95_smoothed < latency_slo_ms * e2e_ok_ratio
  - else if queue budget is set: queue_p95_smoothed < queue_budget_ms

If a rejection occurs in a tick, congestion is set immediately and the entry
horizon is considered satisfied.

Horizon handling:

- When not congested, the entry accumulator adds the elapsed tick duration
  while the entry condition stays true and resets to zero when it becomes
  false. Congestion starts once it reaches `entry_horizon_ms`.
- When congested, the exit accumulator adds elapsed tick duration while the
  exit condition stays true and resets to zero when it becomes false. The
  monitor clears congestion once it reaches `exit_horizon_ms`.

## Congestion score

The `inference_congestion_score` gauge is a normalized score in [0,1] that
tracks how close the system is to congestion. It is computed as the maximum of
three pressure scores, each clamped to [0,1]:

- clamp(x) = min(max(x, 0), 1)
- queue_pressure_score = clamp((fill_smoothed - fill_low) /
  (fill_high - fill_low)) if fill_high > fill_low, else 0.
- capacity_pressure_score = clamp((rho_smoothed - rho_low) /
  (rho_high - rho_low)) if rho_high > rho_low, else 0.
- latency_pressure_score:
  - if SLO is enabled and e2e_p95_smoothed exists:
    - lower = latency_slo_ms * e2e_ok_ratio
    - upper = latency_slo_ms * 1.1
    - score = clamp((e2e_p95_smoothed - lower) / (upper - lower))
  - else if queue_budget_ms is set and queue_p95_smoothed exists:
    - lower = queue_budget_ms
    - upper = queue_budget_ms * 1.2
    - score = clamp((queue_p95_smoothed - lower) / (upper - lower))
  - else: 0

Final score:

- congestion_score = max(queue_pressure_score, latency_pressure_score,
  capacity_pressure_score)

## Configuration fields

All fields live under the `congestion:` block. The block is optional, if it is
omitted, defaults from `RuntimeConfig::CongestionSettings` are used and
congestion detection stays enabled. Any field left out keeps its default value.

- `enabled` (bool): Enable congestion detection.
- `latency_slo_ms` (ms): End-to-end SLO target. If 0, SLO-based checks are off.
- `queue_latency_budget_ms` (ms): Explicit queue latency budget. If 0, the
  ratio below is used to derive a budget from `latency_slo_ms`.
- `queue_latency_budget_ratio` (0..1): Fraction of SLO reserved for queue
  latency when `queue_latency_budget_ms == 0`.
- `e2e_warn_ratio` (0..1): Fraction of `latency_slo_ms` that triggers
  congestion entry when using E2E latency.
- `e2e_ok_ratio` (0..1): Fraction of `latency_slo_ms` required to clear
  congestion when using E2E latency.
- `fill_high` (0..1): Queue fill ratio to enter congestion.
- `fill_low` (0..1): Queue fill ratio to exit congestion (hysteresis).
- `rho_high` (>0): Load threshold to enter congestion.
- `rho_low` (>=0): Load threshold to exit congestion (hysteresis).
- `alpha_ewma` (0..1]: EWMA smoothing factor (higher = more reactive).
- `entry_horizon_ms` (>0): Time window that entry condition must hold before
  entering congestion.
- `exit_horizon_ms` (>0): Time window that exit condition must hold before
  exiting congestion.
- `tick_interval_ms` (>0): Sampling period for the congestion monitor.

## Example YAML

```yaml
congestion:
  enabled: true
  latency_slo_ms: 150
  queue_latency_budget_ms: 30
  fill_high: 0.8
  fill_low: 0.6
  rho_high: 1.05
  rho_low: 0.9
  alpha_ewma: 0.2
  entry_horizon_ms: 5000
  exit_horizon_ms: 15000
  tick_interval_ms: 1000
```

## Outputs and observability

- Prometheus: `inference_congestion_flag`, `inference_congestion_score`,
  `inference_congestion_*` gauges.
- Perfetto trace: `perfetto_trace.json` contains a `congestion` track with
  during congested periods.
- Trace summary: `trace.csv` has a `congested` column for each batch.
