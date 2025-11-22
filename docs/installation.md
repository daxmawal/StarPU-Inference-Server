# StarPU Inference Server - Installation Guide

| [Installation](./installation.md) | [Quickstart](./quickstart.md) | [Server Configuration](./server_guide.md) | [Client Guide](./client_guide.md) | [Docker Guide](./docker_guide.md) |
| --- | --- | --- | --- | --- |

## Installation Guide

This guide explains how to install and build **StarPU Inference Server** natively
on a Linux host.

## Tested environment

- Ubuntu 22.04 LTS
- CUDA 11.8
- GCC 13
- LibTorch 2.2.2 (cu118)

## Hardware and software requirements

- NVIDIA GPU with a driver that supports CUDA 11.8.
- Optional: NVML headers (`libnvidia-ml-dev`) to enable GPU metrics.

> **Note:** the commands below assume Ubuntu 22.04. Adapt package names if you
> are using another distribution.

---

## 1. Prepare the environment

Install dependencies into a dedicated prefix to keep the system clean. If your
StarPU/LibTorch installs already live in standard prefixes, you can skip these
exports and just pass their paths via `CMAKE_PREFIX_PATH` at configure time:

```bash
export INSTALL_DIR="$HOME/Install"
export STARPU_DIR="$INSTALL_DIR/starpu"
export CMAKE_PREFIX_PATH="$STARPU_DIR:$INSTALL_DIR/libtorch"
export LD_LIBRARY_PATH="$INSTALL_DIR/libtorch/lib:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$INSTALL_DIR"
```

Append these exports to `~/.bashrc` so they are available in future sessions.

## 2. Install system packages

```bash
sudo apt-get update
sudo apt-get install -y \
  autoconf automake build-essential git pkg-config \
  libfxt-dev libhwloc-dev libltdl-dev libssl-dev \
  libtool libtool-bin m4 ninja-build unzip zlib1g-dev
```

For GPU telemetry via NVML, also install:

```bash
sudo apt-get install -y nvidia-cuda-toolkit libnvidia-ml-dev
```

## 3. Install GCC 13

```bash
sudo apt-get update
sudo apt-get install -y g++-13
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100
sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-13 100
```

## 4. Install CMake >= 3.28

```bash
CMAKE_VERSION=3.28.3
wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz -O /tmp/cmake.tar.gz
sudo tar -C /usr/local --strip-components=1 -xz -f /tmp/cmake.tar.gz
rm /tmp/cmake.tar.gz
cmake --version
```

## 5. Install LibTorch (cu118)

```bash
LIBTORCH_VERSION=2.2.2
LIBTORCH_CUDA=cu118
wget https://download.pytorch.org/libtorch/${LIBTORCH_CUDA}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_CUDA}.zip \
  -O /tmp/libtorch.zip
unzip /tmp/libtorch.zip -d "$INSTALL_DIR"
rm /tmp/libtorch.zip
```

Verify that `"$INSTALL_DIR/libtorch/lib"` is present in `LD_LIBRARY_PATH`.

## 6. Submodule C++ dependencies (Protobuf/gRPC/Abseil/utf8_range)

These dependencies are vendored as git submodules under `external/` and built as
part of the project when `USE_BUNDLED_DEPS=ON` (default). Make sure submodules
are present:

```bash
git submodule update --init --recursive
```

### StarPU 1.4.8

```bash
STARPU_VERSION=1.4.8
mkdir -p /tmp/starpu
wget https://gitlab.inria.fr/starpu/starpu/-/archive/starpu-${STARPU_VERSION}/starpu-starpu-${STARPU_VERSION}.tar.gz -O /tmp/starpu.tar.gz
tar -xzf /tmp/starpu.tar.gz -C /tmp/starpu --strip-components=1
pushd /tmp/starpu
./autogen.sh
./configure \
  --prefix="$STARPU_DIR" \
  --enable-tracing \
  --with-fxt \
  --disable-hip \
  --disable-opencl \
  --disable-mpi \
  --enable-cuda \
  --disable-fortran \
  --disable-openmp
make -j"$(nproc)"
make install
popd
rm -rf /tmp/starpu /tmp/starpu.tar.gz
```

## 7. Build StarPU Inference Server

Clone the repository (with submodules) :

```bash
git clone --recurse-submodules https://github.com/daxmawal/StarPU-Inference-Server.git
cd StarPU-Inference-Server
# If you already cloned without submodules:
git submodule update --init --recursive
```

Configure and compile:

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

The main executables are emitted under `build/`:

- `starpu_server`: gRPC service combining StarPU and LibTorch.
- `client_example`: sample CLI client.

## 8. Optional: build and run tests

```bash
cmake -S . -B build \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

## CMake options

Pass options at configure time with `-D<OPTION>=<ON|OFF>`.

| Option | Description | Default |
| --- | --- | --- |
| `USE_BUNDLED_DEPS` | Build vendored Protobuf/gRPC/Abseil and friends from `external/` instead of relying on system packages. | ON |
| `BUILD_TESTS` | Build the unit tests and test helpers. | OFF |
| `ENABLE_SANITIZERS` | Enable AddressSanitizer and UBSan. | OFF |
| `ENABLE_COVERAGE` | Enable coverage instrumentation (gcov/lcov). | OFF |
