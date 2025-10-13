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

#### Goal

- Perform inference of TorchScript models (e.g., ResNet, BERT) using LibTorch.
- Dynamically schedule inference tasks between CPU and GPU using StarPU.
- Optimize **throughput** while satisfying **latency constraints**.

##### Features

- [x] Submission of CPU/GPU inference tasks through StarPU
- [x] TorchScript model execution with LibTorch
- [x] Asynchronous execution with custom user callbacks
- [ ] Multithreaded server to receive inference requests
- [ ] Dynamic batching algorithm for improved throughput

## Documentation

The project currently provides three entry points for documentation:

- [Installation Guide](docs/installation.md) – toolchain and dependency setup.
- [Configuration Reference](docs/configuration.md) – YAML schema and
  runtime options.
- [Usage Guide](docs/usage.md) – running the server and using the CLI.

## Quick Start

1. Follow the [Installation Guide](docs/installation.md) to install
   StarPU, LibTorch, gRPC, and the remaining dependencies.
2. Create a configuration file describing your TorchScript model inputs
   and outputs. See the [Configuration Reference](docs/configuration.md)
   for all available options.
3. Build and run the server:

   ```bash
   cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch
   cmake --build build
   ./build/starpu_server --config path/to/config.yaml
   ```

4. Point your gRPC client to the configured address (defaults to
   `0.0.0.0:50051`) and monitor Prometheus metrics on port 9090.

[ci-badge]:
  https://github.com/daxmawal/StarPU-Inference-Server/actions/workflows/ci.yml/badge.svg
[ci-url]:
  https://github.com/daxmawal/StarPU-Inference-Server/actions/workflows/ci.yml
[codecov-badge]:
  https://codecov.io/github/daxmawal/StarPU-Inference-Server/graph/badge.svg?token=WV7HQ2N4T6
[codecov-url]:
  https://codecov.io/github/daxmawal/StarPU-Inference-Server
<!--[sonar-badge]:
  https://sonarcloud.io/images/project_badges/sonarcloud-dark.svg
[sonar-url]:
  https://sonarcloud.io/summary/new_code?id=daxmawal_StarPU-Inference-Server-->
