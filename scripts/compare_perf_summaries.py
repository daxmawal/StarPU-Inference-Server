#!/usr/bin/env python3
"""Compare a candidate perf summary against a baseline summary."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare candidate perf metrics against a baseline with relative "
            "regression tolerances."
        )
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--latency-metric", default="server_overall")
    parser.add_argument("--max-latency-regression-pct", required=True, type=float)
    parser.add_argument("--max-throughput-drop-pct", required=True, type=float)
    parser.add_argument("--max-rejected", type=int, default=0)
    parser.add_argument("--expected-requests", type=int)
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"[perf-compare] {message}", file=sys.stderr)
    raise SystemExit(1)


def require_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)):
        fail(f"{label} is missing or not numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        fail(f"{label} is not finite")
    return numeric


def require_int(value: object, label: str) -> int:
    if not isinstance(value, int):
        fail(f"{label} is missing or not an integer")
    return value


def load_summary(
    path: Path, latency_metric_name: str, expected_requests: int | None, max_rejected: int
) -> tuple[int, int, float, float]:
    if not path.is_file():
        fail(f"summary file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    requests = summary.get("requests")
    if not isinstance(requests, dict):
        fail(f"requests section is missing in {path}")

    sent = require_int(requests.get("sent"), f"{path}: requests.sent")
    handled = require_int(requests.get("handled"), f"{path}: requests.handled")
    rejected = require_int(requests.get("rejected"), f"{path}: requests.rejected")
    throughput = require_number(summary.get("throughput_rps"), f"{path}: throughput_rps")

    latencies = summary.get("latency_ms")
    if not isinstance(latencies, dict):
        fail(f"latency_ms section is missing in {path}")

    latency_metric = latencies.get(latency_metric_name)
    if not isinstance(latency_metric, dict):
        fail(f"latency metric '{latency_metric_name}' is missing in {path}")
    latency_p95 = require_number(
        latency_metric.get("p95_ms"),
        f"{path}: latency_ms.{latency_metric_name}.p95_ms",
    )

    if expected_requests is not None and sent != expected_requests:
        fail(f"{path}: expected {expected_requests} sent requests, got {sent}")
    if handled != sent:
        fail(f"{path}: handled requests ({handled}) do not match sent requests ({sent})")
    if rejected > max_rejected:
        fail(f"{path}: rejected requests {rejected} exceed limit {max_rejected}")
    if throughput <= 0.0:
        fail(f"{path}: throughput must be strictly positive")
    if latency_p95 <= 0.0:
        fail(f"{path}: p95 latency must be strictly positive")

    return sent, rejected, throughput, latency_p95


def main() -> int:
    args = parse_args()

    baseline_sent, baseline_rejected, baseline_throughput, baseline_latency_p95 = (
        load_summary(
            args.baseline,
            args.latency_metric,
            args.expected_requests,
            args.max_rejected,
        )
    )
    (
        candidate_sent,
        candidate_rejected,
        candidate_throughput,
        candidate_latency_p95,
    ) = load_summary(
        args.candidate,
        args.latency_metric,
        args.expected_requests,
        args.max_rejected,
    )

    if candidate_sent != baseline_sent:
        fail(
            f"candidate sent requests ({candidate_sent}) do not match baseline ({baseline_sent})"
        )

    throughput_drop_pct = max(
        0.0, (baseline_throughput - candidate_throughput) / baseline_throughput * 100.0
    )
    latency_regression_pct = max(
        0.0,
        (candidate_latency_p95 - baseline_latency_p95) / baseline_latency_p95 * 100.0,
    )

    print(
        "[perf-compare] "
        f"baseline_throughput_rps={baseline_throughput:.3f} "
        f"candidate_throughput_rps={candidate_throughput:.3f} "
        f"throughput_drop_pct={throughput_drop_pct:.3f} "
        f"baseline_{args.latency_metric}_p95_ms={baseline_latency_p95:.3f} "
        f"candidate_{args.latency_metric}_p95_ms={candidate_latency_p95:.3f} "
        f"latency_regression_pct={latency_regression_pct:.3f} "
        f"baseline_rejected={baseline_rejected} "
        f"candidate_rejected={candidate_rejected}"
    )

    if throughput_drop_pct > args.max_throughput_drop_pct:
        fail(
            "throughput drop "
            f"{throughput_drop_pct:.3f}% exceeds limit "
            f"{args.max_throughput_drop_pct:.3f}%"
        )
    if latency_regression_pct > args.max_latency_regression_pct:
        fail(
            "latency regression "
            f"{latency_regression_pct:.3f}% exceeds limit "
            f"{args.max_latency_regression_pct:.3f}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
