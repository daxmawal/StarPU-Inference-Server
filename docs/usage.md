<!--# Usage Guide

This document explains how to run the StarPU Inference Server after
building it.

## Quick Start

1. Create a configuration file (see [Configuration](configuration.md)).
2. Run the server and point it to the config file:

   ```bash
   ./build/starpu_server --config configs/resnet50.yaml
   ```

3. Submit requests via your gRPC client. The server logs runtime
   information and exposes Prometheus metrics on the configured port.

## Command-Line Interface

The executable accepts numerous options that can be combined with, or
without, a configuration file. Run `starpu_server --help` to see the full
list generated from `get_help_message` in the CLI parser.

Common flags:

- `--config PATH`: Load YAML configuration before parsing other flags.
- `--model PATH`: Path to a TorchScript module (overrides YAML).
- `--scheduler NAME`: Select the StarPU scheduler (e.g. `lws`, `dmda`,
  `pheft`). Unsupported values are rejected by the CLI/parser.
- `--request-number N`: Total number of inference requests to submit.
- `--input name:DIMS:TYPE`: Describe input tensors directly on the
  command line. Repeat for multiple tensors.
- `--device-ids 0,1`: Enable CUDA execution on the listed GPU IDs.
- `--sync`: Run synchronously for debugging.
- `--delay US`: Microseconds to wait between requests.
- `--address HOST:PORT`: Override the gRPC listen address.
- `--metrics-port PORT`: Change the Prometheus metrics port.
- `--verbose LEVEL`: Verbosity from 0 (silent) to 4 (trace).

The CLI parser merges arguments with values coming from the YAML file.
Command-line values take precedence.

## Runtime Behaviour

At startup the program:

1. Loads the configuration and validates required fields such as `model`,
   `input`, and `output` tensor specs.
2. Initializes the StarPU runtime and, if available, GPU backends.
3. Warms up input slots and pre-generated batches before entering the
   inference loop.
4. Reports throughput statistics on shutdown.

Validation failures (missing keys, invalid tensor shapes, etc.) cause a
fatal log message and a non-zero exit status. Exceptions thrown during
inference or configuration parsing are logged with a descriptive message
before the program exits.

## Observability

- Prometheus metrics: exposed via HTTP on `metrics_port` (default 9090).
- NVML GPU telemetry: enabled automatically when the NVML headers and
  library are available at build time.
- Verbose logging: controlled by `verbosity` or `--verbose`.

Use the metrics endpoint to integrate with monitoring dashboards such as
Prometheus + Grafana.-->
