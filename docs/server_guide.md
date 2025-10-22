# StarPU Inference Server Documentation

| [Installation](./installation.md) | [Quickstart](./quickstart.md) | [Server Configuration](./server_guide.md) |
| --- | --- | --- |

## Server Guide

This guide walks through launching the gRPC inference server and crafting the
YAML configuration files it consumes. It assumes you already followed
[installation](docs/installation.md) to install dependencies and build the project.

## 1. Build the server

From the project root:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j"$(nproc)"
```

- `grpc_server` is the executable that exposes the gRPC API.
- `grpc_client_example` is an example client useful for smoke tests.
- To write your own client in C++, Python, etc, see client_guide.md.

## 2. Prepare a model configuration

The server loads exactly one TorchScript model per configuration file. The
configuration is written in YAML and must include the following required keys:

| Key | Description |
| --- | --- |
| `model` | Absolute or relative path to the TorchScript `.pt`/`.ts` file. |
| `inputs` | Sequence describing each input tensor; every element must define `name`, `data_type`, and `dims`. |
| `outputs` | Sequence describing each output tensor; every element must define `name`, `data_type`, and `dims`. |
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
| `use_cpu` | Enable CPU workers. Combine with `use_cuda` for heterogeneous (CPU+GPU) execution. | `true` |
| `use_cuda` | Enable GPU workers. Accepts either `false` or a sequence of mappings such as `[{ device_ids: [0,1] }]`. | `false` |
| `address` | gRPC listen address (host:port). | `127.0.0.1:50051` |
| `metrics_port` | Port for the Prometheus metrics endpoint. | `9100` |

Behavior of `use_cpu` and `use_cuda`:

- `use_cpu: true`, `use_cuda: [{ device_ids: [...] }]` → StarPU runs heterogeneously on CPU and GPU workers.
- `use_cuda: false` or omitted → pipeline runs on CPU workers only (unless the CLI overrides the setting).
- `use_cpu: false`, `use_cuda: [{ ... }]` → pipeline runs on GPU workers only.
- Both `use_cpu: false` and `use_cuda: false` (or an empty sequence) → configuration is invalid; at least one execution backend must be enabled.

Optional keys for debugging:

| Key | Description | Default |
| --- | --- | --- |
| `verbosity` | Log verbosity level. Supported aliases: `0`/`silent`, `1`/`info`, `2`/`stats`, `3`/`debug`, `4`/`trace`. | `0` |
| `dynamic_batching` | Enable dynamic batching (`true`/`false`). | `true` |
| `sync` | Run the StarPU worker pool in synchronous mode (`true`/`false`). | `false` |

## 3. Example: `models/bert.yml`

The repository ships a ready-to-run configuration:

```yaml
scheduler: lws
model: ../models/bert_libtorch.pt
inputs:
  - { name: "input_ids", data_type: "TYPE_INT64", dims: [1, 128] }
  - { name: "attention_mask", data_type: "TYPE_INT64", dims: [1, 128] }
outputs:
  - { name: "output0", data_type: "TYPE_FP32", dims: [1, 128, 768] }
verbosity: 4
address: 127.0.0.1:50051
metrics_port: 9100
max_batch_size: 32
batch_coalesce_timeout_ms: 1000
dynamic_batching: true
sync: false
use_cpu: true
use_cuda:
  - { device_ids: [0] }
pool_size: 12
```

Update `model:` to match the absolute path of your TorchScript model and adjust
the tensor shapes to the sequence length and hidden size exported by your
training pipeline. **The sample assumes batches of size 1 and lets the runtime expand
to `max_batch_size`.**

## 4. Launch the inference server

Once the configuration YAML is ready and the server binaries are built:

```bash
./grpc_server --config path/to/config.yml
```
