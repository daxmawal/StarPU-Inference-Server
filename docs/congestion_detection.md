# Congestion detection

This document explains the congestion detector, its signals, and the
configuration fields in the `congestion:` YAML block.

## Summary

- A background monitor samples queue and latency metrics every tick.
- Metrics are smoothed with an EWMA to reduce noise.
- Congestion enters only after the entry horizon is satisfied.
- Congestion exits only after the exit horizon is satisfied.
- Any rejection in a tick triggers an immediate congestion state.

## Signals and derived metrics

- Arrival rate (lambda): requests/s over the last tick.
- Completion rate (mu): logical jobs/s over the last tick.
- Load (rho): lambda / mu (smoothed).
- Queue fill ratio: queue_size / queue_capacity (smoothed).
- Queue growth rate: d(queue_size)/dt (smoothed).
- Queue latency p95: p95 of queue latency samples (smoothed).
- E2E latency p95: p95 of end-to-end latency samples (smoothed).

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

## Configuration fields

All fields live under the `congestion:` block.

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

Compatibility: `entry_horizon_seconds` and `exit_horizon_seconds` are still
accepted in YAML and converted to milliseconds, but `*_ms` is preferred.

## Example YAML

```yaml
congestion:
  enabled: true
  latency_slo_ms: 150
  queue_latency_budget_ratio: 0.25
  e2e_warn_ratio: 0.90
  e2e_ok_ratio: 0.80
  fill_high: 0.85
  fill_low: 0.65
  rho_high: 1.10
  rho_low: 0.90
  alpha_ewma: 0.4
  entry_horizon_ms: 3000
  exit_horizon_ms: 7000
  tick_interval_ms: 500
```

## Outputs and observability

- Prometheus: `inference_congestion_flag`, `inference_congestion_score`,
  `inference_congestion_*` gauges.
- Perfetto trace: `perfetto_trace.json` contains a `congestion` track with
  red slices during congested periods.
- Trace summary: `trace.csv` has a `congested` column for each batch.

## References

See `docs/congestion_bibliography.md`.
