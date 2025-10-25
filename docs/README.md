# StarPU Inference Server Docs

This directory collects the documentation needed to install, run, and tune the StarPU Inference Server.

| [Installation](./installation.md) | [Quickstart](./quickstart.md) | [Server Configuration](./server_guide.md) | [Client Guide](./client_guide.md) |
| --- | --- | --- | --- |

## Getting Started

- Start with [installation](./installation.md) for building the server directly on your machine. Or see [docker guide](./docker_guide.md) for Docker image build.
- Once installed, [quickstart](./quickstart.md) walks through running inference workloads with the provided tooling.

## Configuration Reference

- Use [server configuration](./server_guide.md) to review runtime options, environment variables, and parameters.

## Client Guide: Examples & Custom Implementations

Client examples:

- `grpc_client_example` – sample C++ CLI that drives the gRPC service, sources under `src/grpc/client`.
- `client/bert_inference_client.py` – Python gRPC client that tokenises text, [Client Guide](./client_guide.md)).

### Backends

- [StarPU](https://starpu.gitlabpages.inria.fr/) – is an open-source runtime system that schedules and manages data for task-based applications across heterogeneous processors (CPUs and GPUs) to maximize performance and portability.
- [LibTorch](https://pytorch.org/cppdocs/) – is PyTorch’s official C++ library, providing high-performance tensors, automatic differentiation, and neural-network APIs for training and deploying deep-learning models in C++.
