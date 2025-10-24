# StarPU Inference Server Documentation

| [Installation](./installation.md) | [Quickstart](./quickstart.md) | [Server Configuration](./server_guide.md) | [Client Guide](./client_guide.md) |
| --- | --- | --- | --- |

## Quickstart Guide

Spin up **StarPU Inference Server** with the `bert-base-uncased` model in four
steps: build the binaries, export the TorchScript model, launch the server with
`models/bert.yml`, and exercise the gRPC API with the provided Python client.

---

## 1. Build the server

Install dependencies following [installation](./installation.md), then compile the
project:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j"$(nproc)"
```

The binaries, including `grpc_server`, are produced inside `build/`.

> Prefer containers? Build the image described in the [docker guide](./docker_guide.md) and run

---

## 2. Export the BERT TorchScript model

Create a lightweight Python environment, install the dependencies, and run the
export script. It downloads `bert-base-uncased`, traces the model, and stores
the artifact in `../models/bert_libtorch.pt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers

python3 ../models/import_bert-base-uncased.py
deactivate
```

> Tip: the script caches downloads under `.hf_cache_clean/`. Set `HF_HOME` if
> you want to reuse a shared cache directory.

---

## 3. Launch the inference server

The configuration file `models/bert.yml` already points to the exported model.
Start the server from the build repository:

```bash
./grpc_server --config ../models/bert.yml
```

The logs should confirm that the model is loaded and the service is listening
on `127.0.0.1:50051`. The sample configuration enables one CPU worker and one
CUDA worker (device `0`). Adjust the `use_cuda` section if you prefer different
GPU IDs or a CPU-only setup ([Server Configuration](./server_guide.md)).

---

## 4. Launch the client

Open a fresh shell to create a small virtual
environment for the Python client, install its dependencies, and send a real
inference request to the running server.

```bash
cd /path/to/StarPU-Inference-Server
python3 -m venv .venv-client
source .venv-client/bin/activate
pip install --upgrade pip
pip install -r client/requirements.txt

python3 client/bert_inference_client.py \
  --server 127.0.0.1:50051 \
  --text "StarPU orchestrates CPU and GPU to serve this request." \
  --reference-model models/bert_libtorch.pt
```

The client tokenises the sentence, calls the gRPC endpoint, prints a statistical
summary of the returned tensor, and (with `--reference-model`) verifies that the
values match a local TorchScript execution within tight tolerances. Adjust the
text, add more `--text` flags for batched requests, or drop `--reference-model`
if you only need the server response.

---

You now have a fully functional inference server ready for BERT workloads.
