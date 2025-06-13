# Inference Scheduling with StarPU and LibTorch

This project combines [StarPU](https://starpu.gitlabpages.inria.fr/) and [LibTorch](https://pytorch.org/cppdocs/) to efficiently **schedule deep learning inference tasks** across CPUs and GPUs of a compute node.  
The main goal is to **maximize throughput** while maintaining **latency control**, by leveraging asynchronous and heterogeneous execution.

⚠️ Project Status: In Development
This project is currently under active development. There are no releases yet, and the interface or features will change frequently. 

## Goal

- Perform inference of TorchScript models (e.g., ResNet, BERT) using LibTorch.
- Dynamically schedule inference tasks between CPU and GPU using StarPU.
- Optimize **throughput** while satisfying **latency constraints**.

## Features

- [x] Submission of CPU/GPU inference tasks through StarPU
- [x] TorchScript model execution with LibTorch
- [x] Asynchronous execution with custom user callbacks
- [ ] Multithreaded server to receive inference requests
- [ ] Dynamic batching algorithm for improved throughput

## Example Usage

```bash
Coming soon
```
