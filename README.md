# StarPU Inference Server

[![CI][ci-badge]][ci-url]
[![codecov][codecov-badge]][codecov-url]
<!--[![SonarQube Cloud][sonar-badge]][sonar-url]-->

## ⚠️ Project Status: In Development

This project is currently under active development. There are no releases yet,
and the interface or features will change frequently.

### Inference Scheduling with StarPU and LibTorch

This project combines [StarPU](https://starpu.gitlabpages.inria.fr/) and
[LibTorch](https://pytorch.org/cppdocs/) to efficiently **schedule deep learning
inference tasks** across CPUs and GPUs of a compute node. The main goal is to
**maximize throughput** while maintaining **latency control**, by leveraging
asynchronous and heterogeneous execution.

### Goal

- Perform inference of TorchScript models (e.g., ResNet, BERT) using LibTorch.
- Dynamically schedule inference tasks between CPU and GPU using StarPU.
- Optimize **throughput** while satisfying **latency constraints**.

## Installation

See [installation](docs/installation.md) for setup instructions,
including dependency lists, and native build steps. See [docker guide](docs/installation.md) for Docker image build commands and execution.

## Quickstart

Follow the [Quickstart guide](docs/quickstart.md) to:

1. Build the gRPC inference server.
2. Export the `bert-base-uncased` TorchScript model.
3. Launch the server with the provided configuration.
4. Drive it using the Python gRPC client or by authoring your own client.

## Documentation

The documentation index lives in the docs [folder](docs/README.md).

[ci-badge]:
  https://github.com/daxmawal/StarPU-Inference-Server/actions/workflows/ci.yml/badge.svg?branch=main
[ci-url]:
  https://github.com/daxmawal/StarPU-Inference-Server/actions/workflows/ci.yml?query=branch%3Amain
[codecov-badge]:
  https://codecov.io/github/daxmawal/StarPU-Inference-Server/graph/badge.svg?token=WV7HQ2N4T6
[codecov-url]:
  https://codecov.io/github/daxmawal/StarPU-Inference-Server
<!--[sonar-badge]:
  https://sonarcloud.io/images/project_badges/sonarcloud-dark.svg
[sonar-url]:
  https://sonarcloud.io/summary/new_code?id=daxmawal_StarPU-Inference-Server-->
