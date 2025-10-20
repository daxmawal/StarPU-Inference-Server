# StarPU Inference Server Docs

This directory collects the documentation needed to install, run, and tune the StarPU Inference Server.

## Getting Started

- Start with `installation.md` for building the server directly on your machine.
- Prefer containerized setups? Follow `installation-docker.md` for Docker instructions.
- Once installed, `usage.md` walks through running inference workloads with the provided tooling.

## Configuration Reference

- Use `server_guide.md` to review runtime options, environment variables, and parameters.

### Resources

Clients:

- `grpc_client_example` – sample CLI that drives the gRPC service; sources under `src/grpc/client`.

Backends:

- [StarPU](https://starpu.gitlabpages.inria.fr/) – Is an open-source runtime system that schedules and manages data for task-based applications across heterogeneous processors (CPUs and GPUs) to maximize performance and portability.
- [LibTorch](https://pytorch.org/cppdocs/) – Is PyTorch’s official C++ library, providing high-performance tensors, automatic differentiation, and neural-network APIs for training and deploying deep-learning models in C++.
