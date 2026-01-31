# Congestion detection

Congestion is a server state where incoming work is outpacing processing
capacity, causing queues and/or tail latencies to grow.
This document explains the congestion detector, its signals, and the
configuration fields in the `congestion:` YAML block.

## Summary

- A background monitor samples queue and latency metrics every tick (default
  1000 ms, configurable via `tick_interval_ms`).
- Metrics are smoothed with an Exponential Weighted Moving Average (EWMA) to reduce noise.
- Congestion enters only after the entry horizon is satisfied.
- Congestion exits only after the exit horizon is satisfied.
- Any rejection in a tick triggers an immediate congestion state.

## Signals and derived metrics

- Arrival rate (λ): requests/s over the last tick.
- Completion rate (μ): logical jobs/s over the last tick.
- Load (ρ): smoothed arrival/processing ratio.
- Queue fill ratio ($f$): queue utilization.
- Queue growth rate ($\dot{q}$): rate of queue_size change in the tick.
- Queue latency p95 ($Q_{95}$): p95 of queue latency samples in the tick.
- E2E latency p95 ($E_{95}$): p95 of end-to-end latency samples in the tick.

Formulas (per tick):

$$
\begin{aligned}
\Delta t &= \max(\Delta t_{\mathrm{elapsed}}, T_{\mathrm{tick}}) \\
c(x) &= \min(\max(x, 0), 1) \\
\tilde{x}_t &= \alpha x_t + (1 - \alpha)\tilde{x}_{t-1} \\
\lambda &= \frac{N_a}{\Delta t} \\
\mu &= \frac{N_c}{\Delta t}
\end{aligned}
$$

$$
\rho =
\begin{cases}
\lambda / \mu, & \mu > 0 \\
1000, & \mu = 0 \land \lambda > 0 \\
0, & \text{otherwise}
\end{cases}
$$

$$
\begin{aligned}
f &= c(q / C) \\
\dot{q} &= \frac{q - q_{prev}}{\Delta t} \\
Q_{95} &= P_{95}(L_q) \\
E_{95} &= P_{95}(L_e)
\end{aligned}
$$

Notes:

- $\Delta t_{\mathrm{elapsed}}$ is the elapsed time since the previous tick,
  $T_{\mathrm{tick}}$ is `tick_interval_ms`.
- $c(x)$ is the clamp function.
- $\alpha$ is `alpha_ewma` smoothing factor.
- $N_a$ and $N_c$ are arrivals and completions in the tick.
- $q$, $C$, and $q_{prev}$ are queue size, capacity, and previous queue size.
- $L_q$ and $L_e$ are the queue and end-to-end latency samples in the tick.
- $\tilde{x}\_t$ denotes the EWMA-smoothed value of $x$ (e.g.,
  $\tilde{\rho}\_t$, $\tilde{f}\_t$, $\tilde{\dot{q}}\_t$, $\tilde{Q}\_{t,95}$,
  $\tilde{E}\_{t,95}$).
- The first sample initializes the EWMA, if no latency samples arrive in a
  tick, the corresponding smoothed percentile is unchanged.
- `queue_budget_ms` is `queue_latency_budget_ms` when set, otherwise
  `latency_slo_ms * queue_latency_budget_ratio` when SLO is enabled.

## Entry and exit logic (per tick)

Formulas (per tick):

Notation: $U$ = under_provisioned, $P$ = queue_pressure, $D$ = latency_danger,
$K$ = latency_ok, $L_{slo}$ = `latency_slo_ms`, $r_{warn}$ = `e2e_warn_ratio`,
$r_{ok}$ = `e2e_ok_ratio`, $B$ = `queue_budget_ms`, $f_{high}$/$f\_{low}$ =
`fill_high`/`fill_low`, $\rho_{high}$/$\rho\_{low}$ = `rho_high`/`rho_low`.

Entry condition is true if ANY of the following holds:

- under_provisioned (capacity shortfall): $\tilde{\rho}\_t$ > $\rho_{high}$
- queue_pressure: $\tilde{f}\_t$ > $f_{high}$ AND $\tilde{\dot{q}}\_t$ > 0
- latency_danger:
  - if $L_{slo}$ > 0: $\tilde{E}\_{t,95}$ > $L\_{slo} \times r\_{warn}$
  - else if queue budget is set: $\tilde{Q}_{t,95}$ > $B$

Exit condition is true if ALL of the following holds:

- $\tilde{f}\_t$ < $f_{low}$
- $\tilde{\rho}\_t$ < $\rho_{low}$
- latency_ok:
  - if $L\_{slo}$ > 0: $\tilde{E}\_{t,95}$ < $L\_{slo} \times r\_{ok}$
  - else if queue budget is set: $\tilde{Q}_{t,95}$ < $B$

$$
\begin{aligned}
U &= \tilde{\rho}\_t > \rho_{high} \\
P &= \tilde{f}\_t > f_{high} \land \tilde{\dot{q}}\_t > 0
\end{aligned}
$$

$$
D =
\begin{cases}
\tilde{E}_{t,95} > L_{slo} \times r_{warn}, & L_{slo} > 0 \\
\tilde{Q}_{t,95} > B, & B > 0 \\
\text{false}, & \text{otherwise}
\end{cases}
$$

$$
K =
\begin{cases}
\tilde{E}_{t,95} < L_{slo} \times r_{ok}, & L_{slo} > 0 \\
\tilde{Q}_{t,95} < B, & B > 0 \\
\text{true}, & \text{otherwise}
\end{cases}
$$

$$
\begin{aligned}
C_{entry} &= U \lor P \lor D \\
C_{exit} &= \tilde{f}\_t < f_{low} \land \tilde{\rho}\_t < \rho_{low} \land K
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
The score is informational only and does not affect congestion entry/exit.

Formulas:

Notation: $S_q$, $S_c$, and $S_l$ are the queue, capacity, and latency pressure
scores (with $L_{slo}$, $r_{ok}$, and $B$ as defined above).

$$
c(x) = \min(\max(x, 0), 1)
$$

$$
S_q =
\begin{cases}
c\left(\frac{\tilde{f}\_t - f_{low}}{f_{high} - f_{low}}\right), & f_{high} > f_{low} \\
0, & \text{otherwise}
\end{cases}
$$

$$
S_c =
\begin{cases}
c\left(\frac{\tilde{\rho}\_t - \rho_{low}}{\rho_{high} - \rho_{low}}\right), & \rho_{high} > \rho_{low} \\
0, & \text{otherwise}
\end{cases}
$$

$$
S_l =
\begin{cases}
c\left(\frac{\tilde{E}_{t,95} - L_{slo} \, r_{ok}}{1.1 L_{slo} - L_{slo} \times r_{ok}}\right), & L_{slo} > 0 \\
c\left(\frac{\tilde{Q}_{t,95} - B}{1.2 B - B}\right) \times & B > 0 \\
0, & \text{otherwise}
\end{cases}
$$

$$
S = \max(S_q, S_l, S_c)
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
