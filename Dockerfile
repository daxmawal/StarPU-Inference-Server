FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS build-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/root
ENV INSTALL_DIR=${HOME}/Install
ENV STARPU_DIR=${INSTALL_DIR}/starpu
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"
ENV PATH="$INSTALL_DIR/protobuf/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
ENV CMAKE_PREFIX_PATH="$INSTALL_DIR/absl:$INSTALL_DIR/utf8_range:${CMAKE_PREFIX_PATH:-}"

# Create working directories
RUN mkdir -p $INSTALL_DIR $HOME/.cache && \
    apt-get update && apt-get install -y \
    autoconf \
    automake \
    build-essential \
    git \
    libhwloc-dev \
    libltdl-dev \
    libssl-dev \
    libtool \
    libtool-bin \
    m4 \
    ninja-build \
    pkg-config \
    software-properties-common \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# === Install CMake 3.28+ ===
RUN wget -qO- https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz \
    | tar --strip-components=1 -xz -C /usr/local

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
    -DCMAKE_PREFIX_PATH="$INSTALL_DIR/absl" && \
    make && make install && \
    rm -rf /tmp/protobuf

RUN nm -C $INSTALL_DIR/protobuf/lib/libprotoc.a | grep absl || echo "Aucun symbole Abseil trouvé dans libprotoc"

# === Build and install utf8_range ===
RUN git clone https://github.com/protocolbuffers/utf8_range.git /tmp/utf8_range && \
    cd /tmp/utf8_range && mkdir build && cd build && \
    cmake .. \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/utf8_range \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DUTF8_RANGE_ENABLE_TESTING=OFF \
    -DCMAKE_PREFIX_PATH="$INSTALL_DIR/absl" && \
    make && make install && \
    rm -rf /tmp/utf8_range

FROM build-base AS protobuf-checkpoint

# === Build and install gRPC (matching Protobuf v25.3) ===
RUN git clone --branch v1.59.0 https://github.com/grpc/grpc.git /tmp/grpc && \
    cd /tmp/grpc && git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake .. \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/grpc \
    -DgRPC_BUILD_TESTS=OFF && \
    make -j$(nproc) && make install && \
    test -f $INSTALL_DIR/grpc/lib/cmake/grpc/gRPCConfig.cmake && \
    rm -rf /tmp/grpc

# === Build and install StarPU 1.4.8 ===
RUN wget -O /tmp/starpu.tar.gz https://gitlab.inria.fr/starpu/starpu/-/archive/starpu-1.4.8/starpu-starpu-1.4.8.tar.gz && \
    mkdir -p /tmp/starpu && \
    tar -xzf /tmp/starpu.tar.gz -C /tmp/starpu --strip-components=1 && \
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
    rm -rf /tmp/starpu /tmp/starpu.tar.gz

# Copy source code
WORKDIR /app
COPY CMakeLists.txt /app/
COPY src/ /app/src/
COPY cmake/ /app/cmake/

# Build project
WORKDIR /app/build
RUN cmake .. \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_PREFIX_PATH="$INSTALL_DIR/protobuf;$INSTALL_DIR/grpc;$INSTALL_DIR/utf8_range;$STARPU_DIR;$INSTALL_DIR/libtorch;$INSTALL_DIR/absl" \
    -DProtobuf_DIR=$INSTALL_DIR/protobuf/lib/cmake/protobuf \
    -DProtobuf_PROTOC_EXECUTABLE=$INSTALL_DIR/protobuf/bin/protoc \
    -DProtobuf_USE_STATIC_LIBS=ON \
    -DENABLE_COVERAGE=OFF \
    -DENABLE_SANITIZERS=OFF \
    && cmake --build .


# Default command
CMD ["./starpu_server"]
