# StarPU Inference Server Guide

This guide walks through launching the gRPC inference server and crafting the
YAML configuration files it consumes. It assumes you already followed
`docs/installation.md` to install dependencies and build the project.

## 1. Build the server

From the build directory :

```bash
mkdir buil && cd build
cmake ..
make -j"$(nproc)"
```

- `grpc_server` is the executable that exposes the gRPC API.
- `grpc_client_example` is an example CLI client useful for smoke tests.

## 2. Prepare a model configuration

The server loads exactly one TorchScript model per configuration file. The
configuration is written in YAML and must describe at least:

| Key | Description |
| --- | --- |
| `model` | Absolute or relative path to the TorchScript `.pt`/`.ts` file. |
| `input` | Sequence describing each input tensor (name, data type, dims). |
| `output` | Sequence describing each output tensor. |

Common optional keys

| Key | Description |
| --- | --- |
| `scheduler` | StarPU scheduler name (e.g., lws). |
| `device_ids` | GPU device IDs to use (e.g., [0]). |
| `verbosity` | Log verbosity level (integer). |

## 3. Example: `models/bert.yml`

The repository ships a ready-to-run configuration:

```yaml
scheduler: lws
model: ../models/bert_libtorch.pt
device_ids: [0]
input:
  - { name: "input_ids", data_type: "TYPE_INT64", dims: [1, 128] }
  - { name: "attention_mask", data_type: "TYPE_INT64", dims: [1, 128] }
output:
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
training pipeline. The sample assumes batches of size 1 and lets StarPU expand
to `max_batch_size` at runtime.

## 4. Launch the inference server

Once the configuration YAML is ready and the server binaries are built:

```bash
./grpc_server --config path/to/config.yml
```
