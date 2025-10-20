# Docker Installation Guide

This document covers the container-based workflow for **StarPU Inference
Server**. The Docker image ships all native dependencies (StarPU, LibTorch,
Protobuf, gRPC, etc.) so you can run the project without installing toolchains
locally.

## Requirements

- Docker 24 or later with GPU support (`docker info | grep -i nvidia`).
- NVIDIA Container Toolkit (formerly `nvidia-docker2`) configured on the host.
- Internet access during the build stage to download third-party archives.
- Sufficient disk space: the final image is roughly 6 GB.

---

## Build the image

From the repository root:

```bash
docker build -t starpu-inference:dev .
```

The `Dockerfile` exposes several build arguments if you need to override the
default stack (Ubuntu 22.04, CUDA 11.8, LibTorch 2.2.2). For example:

```bash
docker build \
  --build-arg CUDA_VERSION=12.2.0 \
  --build-arg UBUNTU_VERSION=22.04 \
  --build-arg LIBTORCH_VERSION=2.3.0 \
  --build-arg LIBTORCH_CUDA=cu121 \
  -t starpu-inference:cuda12 .
```

---

## Run the gRPC server

The image entrypoint is `/usr/local/bin/grpc_server`. Mount your model artifacts
and configuration file into `/workspace` and start the container with GPU
access:

```bash
docker run --rm -it --gpus all \
  -v /path/to/models:/workspace/models \
  -v /path/to/config.yaml:/workspace/config.yaml \
  starpu-inference:dev \
  --config /workspace/config.yaml
```

- `--gpus all` enables CUDA inside the container.
- Adjust the bind mounts to match where your TorchScript models and YAML config
  live on the host.

If you need to pass additional flags to `grpc_server`, append them after the
image name. Use `--help` to check the available options.

---

## Open an interactive shell

For debugging or development inside the container:

```bash
docker run --rm -it --gpus all \
  --entrypoint /bin/bash \
  starpu-inference:dev
```

This drops you into `/workspace` as the non-root user `appuser`. The prebuilt
binary `grpc_server` is located at `/usr/local/bin/grpc_server`.

---

## Tips

- Rebuild the image whenever you modify C++ sources or CMake files; the build
  stage recompiles the project.
- Use `docker build --target build` if you only want to obtain the compiled
  artifacts from the first stage (for debugging or copying binaries out).
- When iterating on configuration files, prefer mounting them with `-v` so you
  can edit them without rebuilding the image.
