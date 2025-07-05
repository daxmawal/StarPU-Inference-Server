FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS build-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/local/home/jd258565
ENV INSTALL_DIR=${HOME}/Install
ENV STARPU_DIR=${INSTALL_DIR}/starpu
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"
ENV PATH="$INSTALL_DIR/protobuf/bin:$PATH"

# Create working directories
RUN mkdir -p $INSTALL_DIR $HOME/.cache

# Install base dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    autoconf \
    automake \
    libtool \
    libltdl-dev \
    libtool-bin \
    libhwloc-dev \
    m4 \
    pkg-config \
    wget \
    unzip \
    cmake \
    libssl-dev \
    ninja-build \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# === Install GCC 13 and set it as default ===
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && apt-get install -y g++-13 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-13 100

# === Install libtorch ===
RUN mkdir -p $INSTALL_DIR/libtorch && \
    wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.2.2%2Bcu118.zip -O /tmp/libtorch.zip && \
    unzip /tmp/libtorch.zip -d $INSTALL_DIR && \
    rm /tmp/libtorch.zip

# === Install FXT (required for --with-fxt in StarPU) ===
RUN apt-get update && apt-get install -y \
    libfxt-dev \
    && rm -rf /var/lib/apt/lists/*

# === Build and install Abseil ===
RUN git clone -b 20230802.1 https://github.com/abseil/abseil-cpp.git /tmp/abseil && \
    cd /tmp/abseil && mkdir build && cd build && \
    cmake .. \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/absl \
      -DBUILD_SHARED_LIBS=OFF && \
    make && make install && \
    rm -rf /tmp/abseil

# === Build and install Protobuf 25.3 (static) ===
RUN git clone --branch v25.3 https://github.com/protocolbuffers/protobuf.git /tmp/protobuf && \
    cd /tmp/protobuf && \
    git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/protobuf \
      -Dprotobuf_BUILD_SHARED_LIBS=OFF \
      -Dprotobuf_BUILD_TESTS=OFF \
      -Dprotobuf_ABSL_PROVIDER=package \
      -DCMAKE_PREFIX_PATH=$INSTALL_DIR/absl && \
    make && make install && \
    rm -rf /tmp/protobuf

RUN nm -C $INSTALL_DIR/protobuf/lib/libprotoc.a | grep absl || echo "Aucun symbole Abseil trouvé dans libprotoc"

FROM build-base AS protobuf-checkpoint

# === Build and install StarPU 1.4.8 ===
RUN git clone --branch starpu-1.4.8 https://gitlab.inria.fr/starpu/starpu.git /tmp/starpu && \
    cd /tmp/starpu && \
    ./autogen.sh && \
    ./configure \
        --prefix=$STARPU_DIR \
        --enable-tracing \
        --with-fxt \
        --disable-hip \
        --disable-opencl \
        --disable-mpi \
        --enable-cuda \
        --disable-fortran \
        --disable-openmp \
        && make && make install && \
    rm -rf /tmp/starpu

# === Build and install gRPC with Protobuf 25.3 and Abseil ===
RUN git clone -b v1.62.0 --recurse-submodules https://github.com/grpc/grpc.git /tmp/grpc && \
    cd /tmp/grpc && mkdir -p build && cd build && \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/grpc \
      -DCMAKE_PREFIX_PATH="$INSTALL_DIR/protobuf;$INSTALL_DIR/absl" \
      -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DgRPC_PROTOBUF_PROVIDER=package \
      -DgRPC_ABSL_PROVIDER=package \
      -DProtobuf_DIR=$INSTALL_DIR/protobuf/lib/cmake/protobuf \
      -DProtobuf_PROTOC_EXECUTABLE=$INSTALL_DIR/protobuf/bin/protoc \
      -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --parallel && cmake --install . && \
    rm -rf /tmp/grpc

# Copy source code
WORKDIR /app
COPY . /app

# Build project
WORKDIR /app/build
RUN cmake .. \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_PREFIX_PATH="$INSTALL_DIR/libtorch;$INSTALL_DIR/grpc;$STARPU_DIR;$INSTALL_DIR/protobuf;$INSTALL_DIR/absl" \
      -DENABLE_COVERAGE=OFF \
      -DENABLE_SANITIZERS=OFF \
      && cmake --build .

# Default command
CMD ["./starpu_server"]
