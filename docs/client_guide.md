# StarPU Inference Server Documentation

| [Installation](./installation.md) | [Quickstart](./quickstart.md) | [Server Configuration](./server_guide.md) | [Client Guide](./client_guide.md) |
| --- | --- | --- | --- |

## Client Guide

This guide explains how to exercise the gRPC inference API with real data. It
focuses on the BERT example shipped with the repository, but the workflow can be
adapted to any TorchScript model once the tensor names and shapes are known.

---

## 1. Set up the Python client environment

The repository ships a ready-to-run Python script that constructs BERT
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

## 2. Launch the inference server

Build the project and start the server with the provided configuration (see
[Quickstart](./quickstart.md) for the full context):

```bash
./grpc_server --config models/bert.yml
```

When the logs show that the server is listening on `127.0.0.1:50051`, you can
drive it with a client.

---

## 3. Run an inference with natural language input

Use `python_client/bert_inference_client.py` to tokenize one or more sentences,
submit them for inference, and print a short summary of the returned tensor.

```bash
python3 python_client/bert_inference_client.py \
  --server 127.0.0.1:50051 \
  --model-name bert \
  --text "Your evaluation sentence"
```

---

## 5. Validate outputs with a local TorchScript model

If you keep the TorchScript artifact on disk, you can double-check that the
server returns the same tensors as a local forward pass. Pass the path via
`--reference-model`; the client will execute the module on CPU, compare the
outputs element-wise, and report the maximum absolute/relative deviation.

```bash
python3 python_client/bert_inference_client.py \
  --server 127.0.0.1:50051 \
  --model-name bert \
  --text "Your evaluation sentence" \
  --reference-model ../models/bert_libtorch.pt \
  --rtol 1e-3 \
  --atol 1e-5
```

The summary now includes a validation line per output.

---

You now have an end-to-end loop that uses real data to validate the StarPU
Inference Server deployment.
