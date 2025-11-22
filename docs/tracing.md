# Tracing Guide (Perfetto & StarPU FXT)

This guide explains how to generate `batching_trace.json`, understand its
contents, inspect it with the Perfetto UI, and capture StarPU FXT traces.
Enable tracing only during benchmarks or local debugging sessions.

## 1. Enable the batching JSON trace

Tracing is disabled by default. Enable it in the `batching` section of your
model configuration:

```yaml
batching:
  trace_enabled: true
  trace_output: /tmp/  # optional custom path
```

- `trace_enabled` flips the instrumentation on as soon as the server starts.
- `trace_output` is optional and must point to a directory, the server writes
  `batching_trace.json` inside. When omitted the server writes the file in the
  working directory. The same directory also receives
  `batching_trace_summary.csv`, a CSV dump of each batch (worker ID and type,
  batch size, request IDs, etc). Warmup batches are excluded.
  The server automatically runs `scripts/plot_batch_summary.py` at shutdown to
  produce plots.

Each server restart truncates the previous file, so copy the trace elsewhere
before launching another run. Stop the server before opening the trace.

## 2. JSON layout

The output follows the Chrome trace-event format.

![Aperçu Perfetto](images/perfetto_example.png)

Key event types:

- `request_enqueued` (track request enqueued) records each incoming request.
- `batch` (track batch) spans the time requests spend waiting for a dynamic
  batch.
- `batch_build` (track dynamic batching) covers the time spent assembling the
  batch before it is handed off to StarPU. Flow arrows link these slices to the
  worker that eventually executes the batch.
- `batch_submitted` (track batch submitted) is an instant event that ties a
  batch to the worker that will execute it.
- Entries named after correspond to worker lanes.

Warmup requests reuse the same keys with a `warming_` prefix so they can be
filtered out quickly inside Perfetto.

## 3. StarPU FXT traces

For detailed StarPU scheduling timelines, enable FXT tracing via environment
variables (inline in your YAML or in the shell):

```bash
STARPU_FXT_TRACE=1 \
STARPU_FXT_PREFIX=/path/to/trace_dir \
./starpu_server --config models/bert.yml
```

This produces `starpu_<pid>.trace` files under the chosen prefix. Inspect them
with `starpu_fxt_tool` or any StarPU-compatible visualisation tool. FXT traces
complement the batching JSON trace by exposing low-level worker scheduling and
CUDA runtime activity.

## 4. Batch summary plots

The server writes `batching_trace_summary.csv` alongside the JSON trace and
automatically runs `scripts/plot_batch_summary.py` at shutdown to generate
latency scatter plots for CPU/GPU batches. Run it manually to re-plot or to
point at archived traces:

```bash
./scripts/plot_batch_summary.py /path/to/batching_trace_summary.csv --output batching_plots.png
```
