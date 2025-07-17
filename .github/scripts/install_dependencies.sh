#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${HOME}/Install"
STARPU_DIR="${INSTALL_DIR}/starpu"

# Install base packages
sudo apt-get update
sudo apt-get install -y \
    autoconf automake build-essential cmake git \
    libhwloc-dev libltdl-dev libssl-dev libtool libtool-bin \
    m4 ninja-build pkg-config software-properties-common \
    unzip wget libfxt-dev lcov libnvtoolsext-dev

# GCC 13
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y g++-13
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100
sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-13 100

mkdir -p "$INSTALL_DIR" "$HOME/.cache"

# Install libtorch
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.2.2%2Bcu118.zip -O /tmp/libtorch.zip
unzip /tmp/libtorch.zip -d "$INSTALL_DIR"
rm /tmp/libtorch.zip

# Build and install Abseil
git clone -b 20230802.1 https://github.com/abseil/abseil-cpp.git /tmp/abseil
cd /tmp/abseil && mkdir build && cd build
cmake .. -DCMAKE_CXX_STANDARD=17 -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/absl -DBUILD_SHARED_LIBS=OFF
make -j"$(nproc)"
sudo make install
cd / && rm -rf /tmp/abseil

# Build and install Protobuf
git clone --branch v25.3 https://github.com/protocolbuffers/protobuf.git /tmp/protobuf
cd /tmp/protobuf && git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/protobuf -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_ABSL_PROVIDER=package -DCMAKE_PREFIX_PATH=$INSTALL_DIR/absl
make -j"$(nproc)"
sudo make install
cd / && rm -rf /tmp/protobuf

# Build and install StarPU
git clone --branch starpu-1.4.8 https://gitlab.inria.fr/starpu/starpu.git /tmp/starpu
cd /tmp/starpu
./autogen.sh
./configure --prefix=$STARPU_DIR --enable-tracing --with-fxt --disable-hip --disable-opencl --disable-mpi --enable-cuda --disable-fortran --disable-openmp --disable-starpupy
make -j"$(nproc)"
sudo make install
# chmod -R u+w /tmp/starpu/starpupy/src/starpupy.egg-info || true
cd /
sudo rm -rf /tmp/starpu

# Build and install gRPC
git clone -b v1.62.0 --recurse-submodules https://github.com/grpc/grpc.git /tmp/grpc
cd /tmp/grpc && mkdir build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/grpc \
  -DCMAKE_PREFIX_PATH="$INSTALL_DIR/protobuf;$INSTALL_DIR/absl" \
  -DgRPC_INSTALL=ON \
  -DgRPC_BUILD_TESTS=OFF \
  -DgRPC_BUILD_CODEGEN=OFF \
  -DgRPC_BUILD_GRPC_CPP_PLUGIN=OFF \
  -DgRPC_BUILD_CSHARP_EXT=OFF \
  -DgRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF \
  -DgRPC_PROTOBUF_PROVIDER=package \
  -DgRPC_ABSL_PROVIDER=package \
  -DProtobuf_DIR=$INSTALL_DIR/protobuf/lib/cmake/protobuf \
  -DProtobuf_PROTOC_EXECUTABLE=$INSTALL_DIR/protobuf/bin/protoc \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
sudo cmake --install .
cd / && sudo rm -rf /tmp/grpc

# Export environment variables for later steps
{
  echo "STARPU_DIR=$STARPU_DIR"
  echo "CMAKE_PREFIX_PATH=$INSTALL_DIR/libtorch;$INSTALL_DIR/grpc;$STARPU_DIR;$INSTALL_DIR/protobuf;$INSTALL_DIR/absl"
} >> "$GITHUB_ENV"