# Configuration Reference

StarPU Inference Server accepts a YAML configuration file. The CLI parser
merges command-line overrides on top of the values loaded from the file.
This reference lists every supported option and its behaviour.

## File Structure

```yaml
# Minimal template
model: /path/to/model.ts
input:
  - name: input0
    dims: [1, 3, 224, 224]
    data_type: float32
output:
  - name: logits
    dims: [1, 1000]
    data_type: float32
```

The top-level keys must form a mapping. Unknown keys are rejected with a
validation error.

## Required Keys

| Key | Description |
| --- | --- |
| `model` | Filesystem path to a TorchScript module. The loader verifies that the path exists. |
| `input` | Sequence of input tensor specifications. Each entry requires `dims` and `data_type`, with optional `name`. |
| `output` | Sequence of output tensor specifications, same schema as `input`. |

## Tensor Specification

Each tensor entry contains:

- `name` (optional): Defaults to `input{i}` / `output{i}` in CLI mode.
- `dims`: List of positive integers. Up to 8 dimensions are supported.
- `data_type`: Scalar type string accepted by LibTorch (e.g. `float32`,
  `int64`).

The loader validates that `dims` are positive and do not overflow
`size_t`, and that `data_type` is a supported Torch dtype.

## Optional Settings

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `verbose` / `verbosity` | string | `info` | Logging level (`silent`, `error`, `warn`, `info`, `debug`, `trace`). |
| `scheduler` | string | `lws` | Must be one of the StarPU schedulers compiled into the runtime. |
| `request_nb` | integer | `1` | Total number of inference requests. Must be ≥ 0. |
| `device_ids` | list[int] | `[]` | Enables CUDA execution and selects GPU IDs. Setting this implicitly flips `use_cuda` to `true`. |
| `delay` | integer | `0` | Microseconds to wait between requests. Must be ≥ 0. |
| `batch_coalesce_timeout_ms` | integer | `0` | Delay before dispatching a partially filled batch. |
| `address` | string | `0.0.0.0:50051` | gRPC listen address. |
| `metrics_port` | integer | `9090` | Prometheus HTTP port (1–65535). |
| `max_message_bytes` | integer | `32 MiB` | Overrides the computed gRPC message size limit. Must be ≥ 0. |
| `max_batch_size` | integer | `1` | Maximum number of requests per batch. Must be > 0. |
| `dynamic_batching` | bool | `false` | Enables dynamic batching heuristics. |
| `input_slots` | integer | `0` | Number of reusable tensor buffers. Must be > 0 when provided. |
| `pregen_inputs` | integer | `10` | Pregenerated requests stored for replay. Must be > 0. |
| `warmup_pregen_inputs` | integer | `2` | Pregenerated inputs used during warmup. Must be > 0. |
| `warmup_request_nb` | integer | `2` | Requests per CUDA device sent during warmup. Must be ≥ 0. |
| `seed` | integer | _unset_ | Optional RNG seed (≥ 0). |
| `rtol` | float | `1e-3` | Relative tolerance for result validation. |
| `atol` | float | `1e-5` | Absolute tolerance for result validation. |
| `validate_results` | bool | `true` | Disable with `false` to skip output checks. |
| `sync` | bool | `false` | Run the pipeline synchronously. |
| `use_cpu` | bool | `true` | Force CPU execution. |
| `use_cuda` | bool | `false` | Force-enable CUDA without specifying `device_ids`. |

## Advanced Features

- **Dynamic batching:** `dynamic_batching: true` combines incoming
  requests up to `max_batch_size`, waiting up to
  `batch_coalesce_timeout_ms` before dispatch.
- **Input slots:** `input_slots` pre-allocates reusable tensor buffers.
- **Message size guard:** When unspecified, the loader estimates the
  maximum gRPC message size from the tensor shapes and `max_batch_size`.

## Warmup Behaviour

The server can warm up GPU kernels by replaying a small number of
requests before serving live traffic. Configure the following keys:

```yaml
warmup_request_nb: 2
warmup_pregen_inputs: 4
pregen_inputs: 16
```

## Error Handling

Any validation failure sets `valid: false` and logs a descriptive error.
Examples include:

- Missing required keys or unknown options.
- Negative values for counts, delays, or tolerances.
- Tensor dimension overflow or unsupported data types.
- Invalid port numbers or message sizes.

The CLI exits early when `valid` is `false`, preventing the server from
starting with a malformed configuration.
