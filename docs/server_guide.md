# StarPU Inference Server Guide

This guide walks through launching the gRPC inference server and crafting the
YAML configuration files it consumes. It assumes you already followed
`docs/installation.md` to install dependencies and build the project.

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
| `inputs` | Sequence describing each input tensor (name, data type, dims). |
| `outputs` | Sequence describing each output tensor. |

Optional keys unlock batching, logging, and runtime controls:

| Key | Description |
| --- | --- |
| `scheduler` | StarPU scheduler name (e.g., lws, eager, heft). |
| `verbosity` | Log verbosity level (`0` silent .. `4` trace). |
| `device_ids` | GPU device IDs to bind (e.g., `[0,1]`). Enables CUDA when non-empty. |
| `use_cpu` | Force CPU execution (`true`/`false`). |
| `use_cuda` | Force CUDA execution (`true`/`false`). |
| `address` | gRPC listen address (host:port). |
| `metrics_port` | Port for the Prometheus metrics endpoint. |
| `request_nb` | Total inference requests to enqueue before exiting (benchmark helper). |
| `delay_us` | Microseconds to sleep between enqueued requests. |
| `batch_coalesce_timeout_ms` | Milliseconds to wait before flushing a dynamic batch. |
| `max_batch_size` | Upper bound for per-request batch size. |
| `dynamic_batching` | Enable dynamic batching (`true`/`false`). |
| `input_slots` | Number of reusable input buffers to pre-allocate. |
| `max_message_bytes` | gRPC max message size in bytes (auto-computed if omitted). |
| `pregen_inputs` | Random input samples to pre-generate for steady-state benchmarking. |
| `warmup_pregen_inputs` | Random input samples to pre-generate for warmup. |
| `warmup_request_nb` | Inference requests issued per CUDA device during warmup. |
| `seed` | Seed for synthetic input generation (integer >= 0). |
| `rtol` | Relative tolerance used when validating outputs. |
| `atol` | Absolute tolerance used when validating outputs. |
| `validate_results` | Toggle result validation (`true`/`false`). |
| `sync` | Run the StarPU worker pool in synchronous mode (`true`/`false`). |

## 3. Example: `models/bert.yml`

The repository ships a ready-to-run configuration:

```yaml
scheduler: lws
model: ../models/bert_libtorch.pt
device_ids: [0]
inputs:
  - { name: "input_ids", data_type: "TYPE_INT64", dims: [1, 128] }
  - { name: "attention_mask", data_type: "TYPE_INT64", dims: [1, 128] }
outputs:
  - { name: "output0", data_type: "TYPE_FP32", dims: [1, 128, 768] }
verbosity: "4"
address: "0.0.0.0:50051"
metrics_port: 9100
max_batch_size: 32
batch_coalesce_timeout_ms: 1000
dynamic_batching: true
sync: false
use_cpu: true
use_cuda: true
input_slots: 12
```

Update `model:` to match the absolute path of your TorchScript model and adjust
the tensor shapes to the sequence length and hidden size exported by your
training pipeline. **The sample assumes batches of size 1 and lets StarPU expand
to `max_batch_size` at runtime.**

## 4. Launch the inference server

Once the configuration YAML is ready and the server binaries are built:

```bash
./grpc_server --config path/to/config.yml
```
