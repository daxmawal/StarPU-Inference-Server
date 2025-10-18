<!--# Installation Guide

This guide describes how to prepare a development environment for the
StarPU Inference Server.

## Prerequisites

StarPU Inference Server is built with CMake and a modern C++ toolchain.
Before configuring the project, install the following base tools:

- CMake 3.28 or newer (required by `cmake_minimum_required`).
- A C++23-capable compiler (GCC 13+, Clang 16+, or MSVC 19.36+).
- Git and standard build utilities (`build-essential` on Debian/Ubuntu).

### Core Runtime Dependencies

The top-level `CMakeLists.txt` requests several packages via
`find_package`. Install these libraries with your system package manager
or build them from source:

| Library | Purpose | Notes |
| --- | --- | --- |
| [StarPU](https://starpu.gitlabpages.inria.fr/) | Heterogeneous task scheduling runtime | Provides CPU/GPU scheduling backend. |
| [LibTorch](https://pytorch.org/cppdocs/installing.html) | PyTorch C++ API for TorchScript models | Download the prebuilt archive matching your CUDA toolkit (or CPU-only). |
| [Protobuf](https://github.com/protocolbuffers/protobuf) | Serialization layer for the gRPC API | gRPC requires version 3.21+. |
| [gRPC](https://grpc.io/docs/languages/cpp/quickstart/) | Remote procedure call framework | Install with CMake config packages enabled. |
| [Abseil](https://abseil.io/) | Utility library used by gRPC | Available via most package managers. |
| [utf8_range](https://github.com/protocolbuffers/utf8_range) | UTF-8 validation helper used by gRPC | Typically installed with gRPC. |
| CUDA Toolkit (optional) | Enables GPU execution and NVTX tracing | Detected automatically if available. |
| NVIDIA NVML (optional) | GPU telemetry for Prometheus metrics | Optional; metrics degrade gracefully if missing. |

> **Tip:** When using the prebuilt LibTorch archive, pass its `share/cmake`
> directory to CMake through `-DCMAKE_PREFIX_PATH=/path/to/libtorch`.

### Ubuntu 22.04 Example

The following commands install the required system packages on Ubuntu 22.04:

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build git pkg-config \
    libprotobuf-dev protobuf-compiler-grpc libgrpc++-dev \
    libabsl-dev libstarpu-dev libutf8-range-dev
```

Then download the LibTorch archive and extract it, e.g.:

```bash
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.2.0+cu121.zip
```

Adjust the URL for your CUDA version or use the CPU-only package if you
plan to run on CPUs only.

## Configure and Build

1. Configure a build directory:

   ```bash
   cmake -S . -B build -GNinja \
       -DCMAKE_PREFIX_PATH=$PWD/libtorch \
       -DCMAKE_BUILD_TYPE=Release
   ```

2. Build the `starpu_server` executable:

   ```bash
   cmake --build build
   ```

3. (Optional) Build unit tests:

   ```bash
   cmake -S . -B build -GNinja \
       -DCMAKE_PREFIX_PATH=$PWD/libtorch \
       -DCMAKE_BUILD_TYPE=RelWithDebInfo \
       -DBUILD_TESTS=ON
   cmake --build build
   ctest --test-dir build
   ```

   Enabling `BUILD_TESTS` downloads GoogleTest via CMake `FetchContent`.

## Runtime Assets

Prepare the TorchScript model(s) referenced in your configuration file.
They must be accessible on the filesystem when the server starts. The
server validates that model paths exist during startup.

## Next Steps

Continue with the [Usage Guide](usage.md) to learn how to launch the
server and submit inference requests.-->
