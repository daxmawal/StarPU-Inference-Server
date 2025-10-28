# StarPU Inference Server — Docker Guide

| [Installation](./installation.md) | [Quickstart](./quickstart.md) | [Server Configuration](./server_guide.md) | [Client Guide](./client_guide.md) | [Docker Guide](./docker_guide.md) |
| --- | --- | --- | --- | --- |

Run through Docker + NVIDIA setup to build, launch, and validate the StarPU inference server in a reproducible environment.

> **Tested environment:** Ubuntu 22.04 LTS on an NVIDIA GPU host with local sudo access.

---

## 1) Install Docker Engine

```bash
# Dependencies
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Docker GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Docker repo
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release; echo $UBUNTU_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install & enable
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```

Verify Docker works:

```bash
sudo docker run --rm hello-world
```

Optional: use Docker without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker  # or log out/in
docker run --rm hello-world
```

---

## 2) Enable GPU in containers (NVIDIA Container Toolkit)

Ensure your host already has a recent NVIDIA driver and that GPU access works before continuing:

```bash
nvidia-smi
```

```bash
# NVIDIA repo
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb #deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] #' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit nvidia-container-toolkit-base \
                        libnvidia-container1 libnvidia-container-tools
```

### 2.1 Configure Docker with the NVIDIA runtime as default

The toolkit ships `nvidia-ctk`, which updates `daemon.json`:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

### 2.2 Fix the NVIDIA runtime cgroups option

Some installs ship with `no-cgroups = true`, which prevents NVML inside containers. Force it to `false`:

```bash
sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups=false --in-place
```

If your toolkit predates `nvidia-ctk`, ensure `/etc/nvidia-container-runtime/config.toml` exists (copy the sample that ships with the package if needed) and then update it manually:

```bash
CONFIG=/etc/nvidia-container-runtime/config.toml
sudo sed -i 's/^\s*no-cgroups\s*=.*/no-cgroups = false/; t; $a no-cgroups = false' "$CONFIG"
```

Restart Docker and validate GPU visibility:

```bash
sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:13.0.0-runtime-ubuntu22.04 nvidia-smi
```

- GPU output → you are ready.
- `Failed to initialize NVML` → re-run the cgroups fix and reboot if required.

---

## 3) Build the StarPU inference image

From the repository root (where the `Dockerfile` lives):

```bash
docker build --no-cache --pull --network=host -t starpu-inference:latest .
```

---

## 4) Prepare model assets and config

Use the repository's `models/` directory (already present at the repo root) and drop in:

- `bert_libtorch.pt` — model weights.
- `bert_docker.yml` — runtime configuration (example below).

If you still need the TorchScript weights, follow the export procedure from [Quickstart – Export the BERT TorchScript model](./quickstart.md#2-export-the-bert-torchscript-model). From the repository root:

```bash
# Adjust the path if your checkout lives elsewhere
cd /path/to/StarPU-Inference-Server

# Reuse an existing venv if you already created it during the quickstart,
# otherwise create a fresh one (remove .venv first if you want a clean slate)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers
python3 models/import_bert-base-uncased.py
deactivate
```

Ensure directories are traversable and files readable:

```bash
chmod 755 models
[ -e models/bert_libtorch.pt ] && chmod 644 models/bert_libtorch.pt
[ -e models/bert_docker.yml ] && chmod 644 models/bert_docker.yml
```

### Example `bert_docker.yml`

> Paths in the YAML refer to container paths. We mount the host directory at `/models`.

```yaml
scheduler: eager
starpu_env:
  STARPU_CUDA_THREAD_PER_WORKER: "1"
  STARPU_CUDA_PIPELINE: "4"
  STARPU_NWORKER_PER_CUDA: "4"
  STARPU_WORKERS_GETBIND: "0"
model: /workspace/models/bert_libtorch.pt
inputs:
  - { name: "input_ids", data_type: "TYPE_INT64", dims: [1, 128] }
  - { name: "attention_mask", data_type: "TYPE_INT64", dims: [1, 128] }
outputs:
  - { name: "output0", data_type: "TYPE_FP32", dims: [1, 128, 768] }
verbosity: 2
address: 127.0.0.1:50051
metrics_port: 9100
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

---

## 5) Run the server container

From the repository root (`/path/to/StarPU-Inference-Server`), start the server
container. Mount the whole workspace so both the configuration file and the
TorchScript artifact land in the expected `/workspace/models` directory inside
the container.

```bash
docker run -d --name starpu-inference \
  --gpus all --network=host \
  --user root \
  -v "$PWD:/workspace:ro" \
  starpu-inference:latest \
  --config /workspace/models/bert_docker.yml
```

> **Note:** The runtime image defaults to an unprivileged `appuser`. If your
> repository tree is not world-readable, grant execute permission on the
> directories (for example `chmod o+rx /path/to/StarPU-Inference-Server{,/models}`)
> or keep `--user root` as shown above so the process can traverse the bind
> mount.

Tail logs:

```bash
docker logs -f --tail=200 starpu-inference
```

---

## 6) Test with the Python gRPC client

With the container running on `--network=host`, you can drive it from your host
using the repository's Python example. Follow the walkthrough in
[Client Guide – Run an inference with natural language input](./client_guide.md#3-run-an-inference-with-natural-language-input)
and, once your virtual environment is ready, send a quick request:

```bash
python client/bert_inference_client.py \
  --server 127.0.0.1:50051 \
  --text "The StarPU server is up and running"
```

The script prints a short summary of the returned tensor (and optionally
validates against a local TorchScript model if you pass `--reference-model`).

---

## 7) Docker Compose (easy relaunch)

Create `docker-compose.yml` in the repository root or alongside your models:

```yaml
services:
  server:
    image: starpu-inference:latest
    gpus: all
    network_mode: host
    user: "root"
    command: ["--config", "/workspace/models/bert_docker.yml"]
    volumes:
      - ./:/workspace:ro
    restart: unless-stopped
```

Run / monitor / stop:

```bash
docker compose up -d
docker compose logs -f
docker compose down
```

> See the permissions note above if you prefer to drop `user: "root"`.

---

**Happy serving!**
