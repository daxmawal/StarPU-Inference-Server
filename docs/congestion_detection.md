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

- Arrival rate (lambda): requests/s over the last tick.
- Completion rate (mu): logical jobs/s over the last tick.
- Load (rho): smoothed arrival/processing ratio.
- Queue fill ratio: smoothed queue utilization.
- Queue growth rate: smoothed rate of queue_size change.
- Queue latency p95: smoothed p95 of queue latency samples in the tick.
- E2E latency p95: smoothed p95 of end-to-end latency samples in the tick.

Formulas (per tick):

$$
\begin{aligned}
dt &= \max(\text{elapsed\_since\_last\_tick}, \text{tick\_interval\_ms}) \\
\operatorname{clamp}(x) &= \min(\max(x, 0), 1) \\
\text{EWMA}_t &= \alpha x_t + (1 - \alpha)\text{EWMA}_{t-1} \\
\lambda &= \frac{\text{arrivals}}{dt} \\
\mu &= \frac{\text{completions}}{dt}
\end{aligned}
$$

$$
\rho_{\text{sample}} =
\begin{cases}
\lambda / \mu, & \mu > 0 \\
1000, & \mu = 0 \text{ and } \lambda > 0 \\
0, & \text{otherwise}
\end{cases}
$$

$$
\begin{aligned}
\rho_{\text{smoothed}} &= \operatorname{EWMA}(\rho_{\text{sample}}) \\
\text{fill\_ratio} &= \operatorname{clamp}\left(\frac{\text{queue\_size}}{\text{queue\_capacity}}\right) \\
\text{fill\_smoothed} &= \operatorname{EWMA}(\text{fill\_ratio}) \\
\text{dqueue} &= \frac{\text{queue\_size} - \text{last\_queue\_size}}{dt} \\
\text{dqueue\_smoothed} &= \operatorname{EWMA}(\text{dqueue}) \\
\text{queue\_p95\_smoothed} &= \operatorname{EWMA}\left(P_{95}(\text{queue\_latency\_samples})\right) \\
\text{e2e\_p95\_smoothed} &= \operatorname{EWMA}\left(P_{95}(\text{e2e\_latency\_samples})\right)
\end{aligned}
$$

Notes:

- dt uses the elapsed time since the previous tick, but never less than
  `tick_interval_ms`.
- The first sample initializes the EWMA. For percentile signals, if no samples
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

Formulas (per tick):

$$
\begin{aligned}
\text{under\_provisioned} &= \rho_{\text{smoothed}} > \rho_{\text{high}} \\
\text{queue\_pressure} &= \text{fill\_smoothed} > \text{fill\_high} \land \text{dqueue\_smoothed} > 0
\end{aligned}
$$

$$
\text{latency\_danger} =
\begin{cases}
\text{e2e\_p95\_smoothed} > \text{latency\_slo\_ms} \cdot \text{e2e\_warn\_ratio}, & \text{latency\_slo\_ms} > 0 \\
\text{queue\_p95\_smoothed} > \text{queue\_budget\_ms}, & \text{queue\_budget\_ms set} \\
\text{false}, & \text{otherwise}
\end{cases}
$$

$$
\text{latency\_ok} =
\begin{cases}
\text{e2e\_p95\_smoothed} < \text{latency\_slo\_ms} \cdot \text{e2e\_ok\_ratio}, & \text{latency\_slo\_ms} > 0 \\
\text{queue\_p95\_smoothed} < \text{queue\_budget\_ms}, & \text{queue\_budget\_ms set} \\
\text{true}, & \text{otherwise}
\end{cases}
$$

$$
\begin{aligned}
\text{entry\_condition} &= \text{under\_provisioned} \lor \text{queue\_pressure} \lor \text{latency\_danger} \\
\text{exit\_condition} &= \text{fill\_smoothed} < \text{fill\_low} \land \rho_{\text{smoothed}} < \rho_{\text{low}} \land \text{latency\_ok}
\end{aligned}
$$

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
three pressure scores, each clamped to [0,1].

Formulas:

$$
\operatorname{clamp}(x) = \min(\max(x, 0), 1)
$$

$$
\text{queue\_pressure\_score} =
\begin{cases}
\operatorname{clamp}\left(\frac{\text{fill\_smoothed} - \text{fill\_low}}{\text{fill\_high} - \text{fill\_low}}\right), & \text{fill\_high} > \text{fill\_low} \\
0, & \text{otherwise}
\end{cases}
$$

$$
\text{capacity\_pressure\_score} =
\begin{cases}
\operatorname{clamp}\left(\frac{\rho_{\text{smoothed}} - \rho_{\text{low}}}{\rho_{\text{high}} - \rho_{\text{low}}}\right), & \rho_{\text{high}} > \rho_{\text{low}} \\
0, & \text{otherwise}
\end{cases}
$$

$$
\text{latency\_pressure\_score} =
\begin{cases}
\operatorname{clamp}\left(\frac{\text{e2e\_p95\_smoothed} - \text{latency\_slo\_ms} \cdot \text{e2e\_ok\_ratio}}{\text{latency\_slo\_ms} \cdot 1.1 - \text{latency\_slo\_ms} \cdot \text{e2e\_ok\_ratio}}\right),
 & \text{latency\_slo\_ms} > 0 \text{ and e2e\_p95\_smoothed exists} \\
\operatorname{clamp}\left(\frac{\text{queue\_p95\_smoothed} - \text{queue\_budget\_ms}}{\text{queue\_budget\_ms} \cdot 1.2 - \text{queue\_budget\_ms}}\right),
 & \text{queue\_budget\_ms set and queue\_p95\_smoothed exists} \\
0, & \text{otherwise}
\end{cases}
$$

$$
\text{congestion\_score} = \max(\text{queue\_pressure\_score}, \text{latency\_pressure\_score}, \text{capacity\_pressure\_score})
$$

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
