# StarPU Inference Server - Server Guide

| [Installation](./installation.md) | [Quickstart](./quickstart.md) | [Server Configuration](./server_guide.md) | [Client Guide](./client_guide.md) | [Docker Guide](./docker_guide.md) |
| --- | --- | --- | --- | --- |

## Server Guide

This guide walks through launching the gRPC inference server and crafting the
YAML configuration files it consumes. It assumes you already followed
[installation](./installation.md) to install dependencies and build the project.

## 1. Build the server

From the project root:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j"$(nproc)"
```

- `starpu_server` is the executable that exposes the gRPC API.
- `client_example` is an example client useful for smoke tests.
- To write your own client in C++, Python, etc, see [Client Guide](./client_guide.md).

## 2. Prepare a model configuration

The server loads exactly one TorchScript model per configuration file. The
configuration is written in YAML and must include the following required keys:

| Key | Description |
| --- | --- |
| `name` | Model configuration identifier. |
| `model` | Absolute or relative path to the TorchScript `.pt`/`.ts` file. |
| `inputs` | Sequence describing each input tensor, every element must define `name`, `data_type`, and `dims`. |
| `outputs` | Sequence describing each output tensor, every element must define `name`, `data_type`, and `dims`. |
| `max_batch_size` | Upper bound for per-request batch size. |
| `batch_coalesce_timeout_ms` | Milliseconds to wait before flushing a dynamic batch. |
| `pool_size` | Number of reusable I/O buffer slots to preallocate per GPU. A value of x allocates x I/O slots per device. |

Each tensor entry under `inputs` or `outputs` must provide:

- `name`: unique identifier.
- `data_type`: tensor element type. Supported values (see `src/proto/model_config.proto`) are:
  - `TYPE_BOOL`
  - `TYPE_UINT8`, `TYPE_UINT16`, `TYPE_UINT32`, `TYPE_UINT64`
  - `TYPE_INT8`, `TYPE_INT16`, `TYPE_INT32`, `TYPE_INT64`
  - `TYPE_FP16`, `TYPE_FP32`, `TYPE_FP64`
  - `TYPE_BF16`
  - `TYPE_STRING`
- `dims`: positive integer dimensions (batch dimension first).

Optional keys unlock batching, logging, and runtime controls:

| Key | Description | Default |
| --- | --- | --- |
| `scheduler` | StarPU scheduler name (e.g., lws, eager, heft). | `lws` |
| `starpu_env` |  Lets you pin StarPU-specific environment variables. | unset |
| `use_cpu` | Enable CPU workers. Combine with `use_cuda` for heterogeneous (CPU+GPU) execution. | `true` |
| `group_cpu_by_numa` | Spawn one StarPU CPU worker per NUMA node instead of per core. | `false` |
| `use_cuda` | Enable GPU workers. Accepts either `false` or a sequence of mappings such as `[{ device_ids: [0,1] }]`. | `false` |
| `address` | gRPC listen address (host:port). | `127.0.0.1:50051` |
| `metrics_port` | Port for the Prometheus metrics endpoint. | `9090` |

Behavior of `use_cpu` and `use_cuda`:

- `use_cpu: true`, `use_cuda: [{ device_ids: [...] }]` → StarPU runs heterogeneously on CPU and GPU workers.
- `use_cuda: false` or omitted → pipeline runs on CPU workers only (unless the CLI overrides the setting).
- `use_cpu: false`, `use_cuda: [{ ... }]` → pipeline runs on GPU workers only.
- Setting `group_cpu_by_numa: true` keeps CPU workers enabled but collapses them to one worker per NUMA node so that each inference shares the full socket instead of a single core.

Optional keys for debugging:

| Key | Description | Default |
| --- | --- | --- |
| `verbosity` | Log verbosity level. Supported aliases: `0`/`silent`, `1`/`info`, `2`/`stats`, `3`/`debug`, `4`/`trace`. | `0` |
| `dynamic_batching` | Enable dynamic batching (`true`/`false`). | `true` |
| `sync` | Run the StarPU worker pool in synchronous mode (`true`/`false`). | `false` |
| `trace_enabled` | Emit batching trace JSON (queueing/assignment/submission/completion events) compatible with the Perfetto UI. | `false` |
| `trace_file` | Output path for the batching Perfetto trace (requires `trace_enabled: true`). | `batching_trace.json` |
| `warmup_batches_per_worker` | Minimum number of full-sized batches each worker executes during the warmup phase. Combined with `max_batch_size` to derive additional warmup requests (set `0` to disable batch-based warmup). | `1` |

Traces use the [Chrome trace-event JSON format](https://perfetto.dev/docs/concepts/trace-formats#json-trace-format), so you can drag the resulting file into [ui.perfetto.dev](https://ui.perfetto.dev) to inspect batching activity. See the [Perfetto trace guide](./perfetto.md) for a step-by-step walkthrough of enabling the trace, interpreting the JSON, and navigating the Perfetto UI. Enable it only while profiling dynamic batching, for detailed StarPU scheduling instrumentation use `STARPU_FXT_TRACE`, and for GPU-wide timelines rely on NVIDIA `nsys`.

During startup the server always schedules a short warmup before accepting real
traffic. The final number of warmup requests is the maximum between the legacy
`warmup_request_nb` (exact request count) and
`warmup_batches_per_worker * max_batch_size`, ensuring each worker executes at
least the configured number of full batches to warm its caches and kernels.

### StarPU environment overrides

The `starpu_env` block lets you pin StarPU-specific environment variables inside
the YAML instead of exporting them in the shell. Each entry is copied into the
process environment before StarPU initialises, so it has the same effect as
`STARPU_*=value ./starpu_server ...`.

```yaml
starpu_env:
  STARPU_CUDA_THREAD_PER_WORKER: "1"
  STARPU_CUDA_PIPELINE: "4"
  STARPU_NWORKER_PER_CUDA: "4"
  STARPU_WORKERS_GETBIND: "0"
```

- `STARPU_CUDA_THREAD_PER_WORKER`: number of CPU helper threads created per CUDA
  worker. A value of `1` keeps one submission thread per StarPU GPU worker.
- `STARPU_CUDA_PIPELINE`: depth of the CUDA pipeline, i.e., the number of
  asynchronous stages StarPU can queue concurrently on each worker.
- `STARPU_NWORKER_PER_CUDA`: number of StarPU workers spawned for each physical
  CUDA device; higher values allow more concurrent CUDA streams per GPU.
- `STARPU_WORKERS_GETBIND`: when set to `0`, disables StarPU’s attempt to query
  and enforce CPU affinities while initialising workers (can help when bindings
  are managed externally).

## 3. Example: `models/bert.yml`

The repository ships a ready-to-run configuration:

```yaml
scheduler: lws
name: bert_local
model: ../models/bert_libtorch.pt
inputs:
  - { name: "input_ids", data_type: "TYPE_INT64", dims: [1, 128] }
  - { name: "attention_mask", data_type: "TYPE_INT64", dims: [1, 128] }
outputs:
  - { name: "output0", data_type: "TYPE_FP32", dims: [1, 128, 768] }
verbosity: 4
address: 127.0.0.1:50051
metrics_port: 9090
max_batch_size: 32
batch_coalesce_timeout_ms: 1000
dynamic_batching: true
sync: false
use_cpu: true
group_cpu_by_numa: true
use_cuda:
  - { device_ids: [0] }
pool_size: 12
```

Pick a distinct `name` for each deployment (e.g., `bert_local`, `bert_prod`) so
logs and metrics identify which configuration is running.

Update `model:` to match the absolute path of your TorchScript model and adjust
the tensor shapes to the sequence length and hidden size exported by your
training pipeline. **The sample assumes batches of size 1 and lets the runtime expand
to `max_batch_size`.**

## 4. Launch the inference server

Once the configuration YAML is ready and the server binaries are built:

```bash
./starpu_server --config path/to/config.yml
```
