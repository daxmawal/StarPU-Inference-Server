# StarPU Inference Server Docs

This directory collects the documentation needed to install, run, and tune the StarPU Inference Server.

| [Installation](./installation.md) | [Quickstart](./quickstart.md) | [Server Configuration](./server_guide.md) | [Client Guide](./client_guide.md) |
| --- | --- | --- | --- |

## Getting Started

- Start with [installation](./installation.md) for building the server directly on your machine.
- Once installed, [quickstart](./quickstart.md) walks through running inference workloads with the provided tooling.

## Configuration Reference

- Use [server configuration](./server_guide.md) to review runtime options, environment variables, and parameters.

## Client Guide: Examples & Custom Implementations

Clients example:

- `grpc_client_example` – sample CLI that drives the gRPC service; sources under `src/grpc/client`.
- `python_client/bert_inference_client.py` – Python gRPC client that tokenises
  real text or replays pre-encoded tensors (see [Client Guide](./client_guide.md)).

Write your own client :

### Backends

- [StarPU](https://starpu.gitlabpages.inria.fr/) – Is an open-source runtime system that schedules and manages data for task-based applications across heterogeneous processors (CPUs and GPUs) to maximize performance and portability.
- [LibTorch](https://pytorch.org/cppdocs/) – Is PyTorch’s official C++ library, providing high-performance tensors, automatic differentiation, and neural-network APIs for training and deploying deep-learning models in C++.
