# StarPU Inference Server — Docker Guide (Ubuntu 22.04 + NVIDIA GPU)

This guide walks you **end‑to‑end** through:

1) installing Docker,  
2) enabling **GPU** support in containers (NVIDIA Container Toolkit),  
3) building the image,  
4) running and testing the inference server,  

> Written for **Ubuntu 22.04** with an **NVIDIA** GPU and `sudo` privileges.

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

# Install
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Optional: use Docker without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

---

## 2) Enable GPU in containers: NVIDIA Container Toolkit

```bash
# NVIDIA repo
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
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

> **Backup** then write a minimal config. If you already have other Docker settings, merge instead of overwriting.

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

Some installs have `no-cgroups = true`, which breaks NVML inside containers. Force it to `false`:

```bash
sudo sed -i 's/^\s*no-cgroups\s*=.*/no-cgroups = false/; t; $a no-cgroups = false' \
  /etc/nvidia-container-runtime/config.toml
```

Restart Docker and **test inside a container** (use a **CUDA 13.0** image, compatible with 58x drivers):

```bash
sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:13.0.0-runtime-ubuntu22.04 nvidia-smi
```

- If you see your GPU → great.
- If you see **“Failed to initialize NVML”**, recheck the cgroups step above and **reboot** if needed.

---

## 3) Build your server image

From the **repo root** (where the `Dockerfile` is):

```bash
docker build --no-cache --pull --network=host -t starpu-inference:latest .
```

Check the binary help:

```bash
docker run --rm --gpus all starpu-inference:latest --help
```

> If the build fails with NVCC 11.8 + GCC 13, force GCC 12 for the CUDA host compiler in your Dockerfile:
>
> ```Dockerfile
> RUN apt-get update && apt-get install -y g++-12
> ENV CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
> # or add -allow-unsupported-compiler to CUDA flags (less clean)
> ```

---

## 4) Prepare the config and models

Create a directory on the host, e.g. `~/Workspace/StarPU-Inference-Server/models/`, containing:

- `bert_libtorch.pt` (weights)
- `bert_docker.yml` (configuration file)

### Example `bert_docker.yml`

> Use **absolute** paths **inside the container**. Ensure files are readable (`chmod 644`) and directories are traversable (`chmod 755`).

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

## 5) Run the server

Change into the **`models/` directory** (so `$PWD` maps to the correct files):

```bash
docker run -d --name starpu-inference \
  --gpus all --network=host \
  -v "$PWD:/workspace/models:ro" \
  starpu-inference:latest \
  --config /workspace/models/bert_docker.yml
```

View logs:

```bash
docker logs -f --tail=200 starpu-inference
```

---

## 6) Test with the bundled gRPC client

From another container:

```bash
docker run --rm --network=host \
  starpu-inference:latest \
  /usr/local/bin/grpc_client_example --address localhost:50051
```

---

## 7) Routine operations

```bash
# Status
docker ps -a --filter name=starpu-inference

# Logs
docker logs -f --tail=200 starpu-inference

# Stop & remove
docker stop starpu-inference
docker rm starpu-inference

# Remove image (optional)
docker rmi starpu-inference:latest
```

---

## 8) Docker Compose (easy relaunch)

Create `docker-compose.yml` (in repo root or in `models/`):

```yaml
services:
  server:
    image: starpu-inference:latest
    gpus: all
    network_mode: host
    command: ["--config", "/workspace/models/bert_docker.yml"]
    volumes:
      - ./models:/workspace/models:ro
```

Run / logs / stop:

```bash
docker compose up -d
docker compose logs -f
docker compose down
```

---

**Happy serving!**
