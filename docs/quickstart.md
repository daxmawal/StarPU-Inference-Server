# StarPU Inference Server Documentation

| [Installation](./installation.md) | [Quickstart](./quickstart.md) | [Server Configuration](./server_guide.md) |
| --- | --- | --- |

## Quickstart Guide

Spin up **StarPU Inference Server** with the `bert-base-uncased` model in three
steps: build the binaries, export the TorchScript model, and launch the server
with `models/bert.yml`.

---

## 1. Build the server

Install dependencies following [installation](docs/installation.md), then compile the
project:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j"$(nproc)"
```

The binaries, including `grpc_server`, are produced inside `build/`.

---

## 2. Export the BERT TorchScript model

Create a lightweight Python environment, install the dependencies, and run the
export script. It downloads `bert-base-uncased`, traces the model, and stores
the artifact in `models/bert_libtorch.pt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers

python ../models/import_bert-base-uncased.py
deactivate
```

> Tip: the script caches downloads under `.hf_cache_clean/`. Set `HF_HOME` if
> you want to reuse a shared cache directory.

---

## 3. Launch the inference server

The configuration file `models/bert.yml` already points to the exported model.
Start the server from the repository root:

```bash
./build/grpc_server --config models/bert.yml
```

The logs should confirm that the model is loaded and the service is listening
on `127.0.0.1:50051`. The sample configuration enables one CPU worker and one
CUDA worker (device `0`). Adjust the `use_cuda` section if you prefer different
GPU IDs or a CPU-only setup.

---

You now have a fully functional inference server ready for BERT workloads.
