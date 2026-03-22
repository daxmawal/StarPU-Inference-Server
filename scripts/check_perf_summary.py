#!/usr/bin/env python3
"""Validate client_example performance summary JSON against CI thresholds."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check throughput and latency thresholds from a perf summary JSON."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--latency-metric", default="server_overall")
    parser.add_argument("--max-latency-p95-ms", required=True, type=float)
    parser.add_argument("--min-throughput-rps", required=True, type=float)
    parser.add_argument("--max-rejected", type=int, default=0)
    parser.add_argument("--expected-requests", type=int)
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"[perf-check] {message}", file=sys.stderr)
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


def main() -> int:
    args = parse_args()
    if not args.summary.is_file():
        fail(f"summary file not found: {args.summary}")

    with args.summary.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    requests = summary.get("requests")
    if not isinstance(requests, dict):
        fail("requests section is missing")

    sent = require_int(requests.get("sent"), "requests.sent")
    handled = require_int(requests.get("handled"), "requests.handled")
    rejected = require_int(requests.get("rejected"), "requests.rejected")
    throughput = require_number(summary.get("throughput_rps"), "throughput_rps")

    latencies = summary.get("latency_ms")
    if not isinstance(latencies, dict):
        fail("latency_ms section is missing")

    latency_metric = latencies.get(args.latency_metric)
    if not isinstance(latency_metric, dict):
        fail(f"latency metric '{args.latency_metric}' is missing")
    latency_p95 = require_number(
        latency_metric.get("p95_ms"), f"latency_ms.{args.latency_metric}.p95_ms"
    )

    print(
        "[perf-check] "
        f"handled={handled} sent={sent} rejected={rejected} "
        f"throughput_rps={throughput:.3f} "
        f"{args.latency_metric}_p95_ms={latency_p95:.3f}"
    )

    if args.expected_requests is not None and sent != args.expected_requests:
        fail(
            f"expected {args.expected_requests} sent requests, got {sent}"
        )
    if handled != sent:
        fail(f"handled requests ({handled}) do not match sent requests ({sent})")
    if rejected > args.max_rejected:
        fail(f"rejected requests {rejected} exceed limit {args.max_rejected}")
    if throughput < args.min_throughput_rps:
        fail(
            f"throughput {throughput:.3f} rps is below minimum {args.min_throughput_rps:.3f} rps"
        )
    if latency_p95 > args.max_latency_p95_ms:
        fail(
            f"p95 latency {latency_p95:.3f} ms exceeds maximum {args.max_latency_p95_ms:.3f} ms"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
