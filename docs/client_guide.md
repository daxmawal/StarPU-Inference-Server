# StarPU Inference Server Documentation

| [Installation](./installation.md) | [Quickstart](./quickstart.md) | [Server Configuration](./server_guide.md) | [Client Guide](./client_guide.md) |
| --- | --- | --- | --- |

## Client Guide

This guide explains how to exercise the gRPC inference API with real data. It
focuses on the BERT example shipped with the repository, but the workflow can be
adapted to any TorchScript model once the tensor names and shapes are known.

---

## 1. Launch the inference server

Build the project and start the server with the provided configuration (see
[Quickstart](./quickstart.md) for the full context):

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j"$(nproc)"
cd ..

./build/grpc_server --config models/bert.yml
```

When the logs show that the server is listening on `127.0.0.1:50051`, you can
drive it with a client.

---

## 2. Set up the Python client environment

The repository ships a ready-to-run Python script that constructs real BERT
inputs (token IDs and attention masks) and sends them through the gRPC API.

```bash
python3 -m venv .venv-client
source .venv-client/bin/activate
pip install --upgrade pip
pip install -r python_client/requirements.txt
```

> The requirements include `grpcio`, `protobuf`, `numpy` and `transformers`.
> Installing them inside a virtual environment keeps the system Python clean.

---

## 3. Run an inference with natural language input

Use `python_client/bert_inference_client.py` to tokenize one or more sentences,
submit them for inference, and print a short summary of the returned tensor.

```bash
python python_client/bert_inference_client.py \
  --server 127.0.0.1:50051 \
  --model-name bert \
  --model-version 1 \
  --text "StarPU orchestrates heterogeneous compute resources." \
  --text "Servir une requête réelle valide la chaîne d'inférence." \
  --print-values 16
```

Output example:

```
Modèle: bert (version: 1)
- Sortie output0: shape=(2, 128, 768) dtype=TYPE_FP32
  min=-7.226942 max=6.871415 mean=0.001243
```

Arguments worth noting:

- `--max-length` adjusts the tokenizer padding/truncation (defaults to 128 to
  match `models/bert.yml`).
- `--output` lets you request additional named outputs if the model exposes
  several tensors.
- `--request-id` attaches a custom identifier so you can correlate the server
  logs with the client request.
- `--print-values` controls how many raw values are printed per output tensor
  (set it to `0` if you only need the statistical summary).
- `--reference-model` feeds the same inputs through a local TorchScript model
  and compares the tensors returned by the server against it. Combine with
  `--rtol`/`--atol` to tune tolerances.

---

## 4. Send pre-encoded dataset samples

If your training or evaluation pipeline already stores tokenized tensors, load
them from a NumPy archive (`.npz`) instead of relying on HuggingFace at runtime.

```python
import numpy as np

# Suppose `sample` comes from your dataset (PyTorch, NumPy, etc.)
np.savez(
    "batch.npz",
    input_ids=sample["input_ids"].astype(np.int64),
    attention_mask=sample["attention_mask"].astype(np.int64),
)
```

Send the file through the client:

```bash
python python_client/bert_inference_client.py \
  --server 127.0.0.1:50051 \
  --model-name bert \
  --encoded-npz batch.npz
```

The script validates that the tensors are rank-2, contiguous, and matching the
shape defined in `models/bert.yml`. Batch dimensions are honoured, which lets
you replay several real samples at once.

---

## 5. Validate outputs with a local TorchScript model

If you keep the TorchScript artifact on disk, you can double-check that the
server returns the same tensors as a local forward pass. Pass the path via
`--reference-model`; the client will execute the module on CPU, compare the
outputs element-wise, and report the maximum absolute/relative deviation.

```bash
python python_client/bert_inference_client.py \
  --server 127.0.0.1:50051 \
  --model-name bert \
  --text "Validation croisée avec le modèle local." \
  --reference-model ../models/bert_libtorch.pt \
  --rtol 1e-3 \
  --atol 1e-5
```

The summary now includes a validation line per output, for example:

```
Validation output0: OK (max abs diff=2.1e-06, max rel diff=4.2e-06)
```

Increase the tolerances if you observe legitimate numerical drift (e.g., CPU vs
GPU execution) or drop the flag to skip the check entirely.

---

## 6. Adapting the workflow to another model

For a different TorchScript model:

1. Update the YAML configuration passed to `grpc_server` so that the tensor
   names (`inputs[i].name`) and shapes (`inputs[i].dims`) match your export.
2. Adjust the client script accordingly:
   - Change the input names passed to `_add_input_tensor`.
   - Serialize tensors with the correct dtype (`TYPE_FP16`, `TYPE_INT8`, etc.).
   - Request the relevant outputs.
3. If the model consumes images or other modalities, replace the tokenizer logic
   with the appropriate preprocessing routine (e.g., OpenCV + normalization).

Because the script relies directly on the generated protobuf stubs, it does not
require any change on the server side—only the request content must align with
the model signature.

---

You now have an end-to-end loop that uses real data to validate the StarPU
Inference Server deployment. Combine it with your regression suite or dataset
samplers to automate correctness checks.
