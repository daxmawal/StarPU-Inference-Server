# StarPU Inference Server — Docker Guide (Ubuntu 22.04 + NVIDIA GPU)

Run through Docker + NVIDIA setup to build, launch, and validate the StarPU inference server in a reproducible environment.

> **Target environment:** Ubuntu 22.04 LTS on an NVIDIA GPU host with local sudo access.

---

## Before you start

- **OS & drivers:** `lsb_release -ds` should report Ubuntu 22.04; `nvidia-smi` must succeed before moving on.
- **Accounts:** a sudo-enabled user (join the `docker` group if you want rootless `docker` later).
- **Network:** outbound HTTPS to `download.docker.com`, `nvidia.github.io`, and Docker Hub.
- **Disk:** ~3 GiB for Docker packages plus the StarPU image and model artifacts.

Quick sanity checks:

```bash
docker --version || true
nvidia-smi || true
```

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

Backup first—merge manually if you already maintain a custom `daemon.json`.

```bash
sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak 2>/dev/null || true

sudo bash -lc 'cat >/etc/docker/daemon.json <<EOF
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": { "path": "nvidia-container-runtime", "runtimeArgs": [] }
  }
}
EOF'
```

### 2.2 Fix the NVIDIA runtime cgroups option

Some installs ship with `no-cgroups = true`, which prevents NVML inside containers. Force it to `false`:

```bash
sudo sed -i 's/^\s*no-cgroups\s*=.*/no-cgroups = false/; t; $a no-cgroups = false' \
  /etc/nvidia-container-runtime/config.toml
```

Restart Docker and validate GPU visibility (CUDA 13.0 runtime works with 535/550 series drivers):

```bash
sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:13.0.0-runtime-ubuntu22.04 nvidia-smi
```

- GPU output ✅ → you are ready.
- `Failed to initialize NVML` → re-run the cgroups fix and reboot if required.

---

## 3) Build the StarPU inference image

From the repository root (where the `Dockerfile` lives):

```bash
docker build --no-cache --pull --network=host -t starpu-inference:latest .
```

---

## 4) Prepare model assets and config

Create a host directory (e.g. `~/Workspace/StarPU-Inference-Server/models/`) and drop in:

- `bert_libtorch.pt` — model weights.
- `bert_docker.yml` — runtime configuration (example below).

Ensure directories are traversable and files readable:

```bash
chmod 755 ~/Workspace/StarPU-Inference-Server/models
chmod 644 ~/Workspace/StarPU-Inference-Server/models/*
```

### Example `bert_docker.yml`

> Paths in the YAML refer to container paths. We mount the host directory at `/workspace/models`.

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
use_cuda:
  - { device_ids: [0] }
pool_size: 12
```

---

## 5) Run the server container

Change into the directory that contains `bert_docker.yml` so the volume mount resolves correctly:

```bash
cd ~/Workspace/StarPU-Inference-Server/models

docker run -d --name starpu-inference \
  --gpus all --network=host \
  -v "$PWD:/workspace/models:ro" \
  starpu-inference:latest \
  --config /workspace/models/bert_docker.yml
```

Tail logs:

```bash
docker logs -f --tail=200 starpu-inference
```

---

## 6) Test with the bundled gRPC client

```bash
docker run --rm --network=host \
  starpu-inference:latest \
  /usr/local/bin/grpc_client_example --address localhost:50051
```

Expect a successful RPC response with timing information.

---

## 7) Routine operations

```bash
# Status / health
docker ps -a --filter name=starpu-inference
docker inspect starpu-inference --format '{{ .State.Status }}'

# Logs
docker logs -f --tail=200 starpu-inference

# Stop & remove container
docker stop starpu-inference
docker rm starpu-inference

# Remove image (optional)
docker rmi starpu-inference:latest
```

---

## 8) Docker Compose (easy relaunch)

Create `docker-compose.yml` in the repository root or alongside your models:

```yaml
services:
  server:
    image: starpu-inference:latest
    gpus: all
    network_mode: host
    command: ["--config", "/workspace/models/bert_docker.yml"]
    volumes:
      - ./models:/workspace/models:ro
    restart: unless-stopped
```

Run / monitor / stop:

```bash
docker compose up -d
docker compose logs -f
docker compose down
```

> Compose v2.6+ understands the `gpus:` stanza. For older Compose, add `deploy.resources.reservations.devices` instead.

---

**Happy serving!**
