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
pip install -r client/requirements.txt
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

Use `client/bert_inference_client.py` to tokenize one or more sentences,
submit them for inference, and print a short summary of the returned tensor.

```bash
python3 client/bert_inference_client.py \
  --server 127.0.0.1:50051 \
  --text "Your evaluation sentence" \
  --text "Add a second sentence for batching" \
  --reference-model ../models/bert_libtorch.pt \
  --rtol 1e-3 \
  --atol 1e-5
```

The script accepts either repeated `--text` flags (batched tokenisation).
Responses contain a single tensor shaped `[batch, 128, 768]` by default; the
client prints sample values and, when `--reference-model` is set, validates the
outputs against a local TorchScript model using the provided tolerances.

## 4. Write your own client

If you need to write your own client, start from these references:

- C++ example: `src/grpc/client/client_main.cpp`.
- Python example: `client/bert_inference_client.py`.
- gRPC service definition: `src/proto/grpc_service.proto` and `src/proto/model_config.proto`.

Generate fresh Python stubs with:

```bash
python3 -m grpc_tools.protoc \
  -I src/proto \
  --python_out=client \
  --grpc_python_out=client \
  src/proto/grpc_service.proto
```

The same proto files underpin the C++ client; CMake handles codegen through
`protobuf_generate` during the normal build.

---

You now have an end-to-end loop that uses real data to validate the StarPU
Inference Server deployment.
