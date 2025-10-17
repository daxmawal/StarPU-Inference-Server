ARG CUDA_VERSION=11.8.0
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/root
ENV INSTALL_DIR=${HOME}/Install
ENV STARPU_DIR=${INSTALL_DIR}/starpu
ENV CMAKE_CUDA_ARCHITECTURES="80;86"
ENV PATH="$INSTALL_DIR/protobuf/bin:$PATH"
ENV LD_LIBRARY_PATH="$INSTALL_DIR/libtorch/lib:$INSTALL_DIR/grpc/lib:$STARPU_DIR/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
ARG CMAKE_PREFIX_PATH=""
ENV CMAKE_PREFIX_PATH="$INSTALL_DIR/absl:$INSTALL_DIR/utf8_range${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

# Create working directories and install build dependencies
RUN mkdir -p "$INSTALL_DIR" "$HOME/.cache" && \
    apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    git \
    libfxt-dev \
    libgtest-dev \
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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# === Install CMake ${CMAKE_VERSION} ===
ARG CMAKE_VERSION=3.28.3
RUN wget -qO- https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz \
    | tar --strip-components=1 -xz -C /usr/local

# === Install GCC ${GCC_VERSION} and set it as default ===
ARG GCC_VERSION=13
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && apt-get install -y --no-install-recommends g++-${GCC_VERSION} && \
    apt-get purge -y --auto-remove software-properties-common && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-${GCC_VERSION} 100

# === Install libtorch ${LIBTORCH_VERSION} (${LIBTORCH_CUDA}) ===
ARG LIBTORCH_VERSION=2.2.2
ARG LIBTORCH_CUDA=cu118
RUN mkdir -p "$INSTALL_DIR/libtorch" && \
    wget https://download.pytorch.org/libtorch/${LIBTORCH_CUDA}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_CUDA}.zip -O /tmp/libtorch.zip && \
    unzip /tmp/libtorch.zip -d "$INSTALL_DIR" && \
    rm /tmp/libtorch.zip

# === Build and install Abseil ===
RUN git clone --depth 1 --branch 20230802.1 https://github.com/abseil/abseil-cpp.git /tmp/abseil && \
    cmake -S /tmp/abseil -B /tmp/abseil/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/absl" \
    -DBUILD_SHARED_LIBS=OFF && \
    cmake --build /tmp/abseil/build -j"$(nproc)" && \
    cmake --install /tmp/abseil/build && \
    rm -rf /tmp/abseil

# === Build and install Protobuf 25.3 (static) ===
RUN git clone --depth 1 --branch v25.3 https://github.com/protocolbuffers/protobuf.git /tmp/protobuf && \
    git -C /tmp/protobuf submodule update --init --recursive && \
    cmake -S /tmp/protobuf -B /tmp/protobuf/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/protobuf" \
    -Dprotobuf_BUILD_SHARED_LIBS=OFF \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_ABSL_PROVIDER=package \
    -DCMAKE_PREFIX_PATH="$INSTALL_DIR/absl" && \
    cmake --build /tmp/protobuf/build -j"$(nproc)" && \
    cmake --install /tmp/protobuf/build && \
    rm -rf /tmp/protobuf

# === Compile GTest ===
RUN cmake -S /usr/src/googletest -B /usr/src/googletest/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /usr/src/googletest/build -j"$(nproc)" && \
    mv /usr/src/googletest/build/lib/*.a /usr/lib && \
    rm -rf /usr/src/googletest

# === Build and install utf8_range ===
RUN git clone --depth 1 --branch main https://github.com/protocolbuffers/utf8_range.git /tmp/utf8_range && \
    cmake -S /tmp/utf8_range -B /tmp/utf8_range/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/utf8_range" \
    -DBUILD_SHARED_LIBS=OFF \
    -Dutf8_range_ENABLE_TESTS=OFF \
    -DBUILD_TESTING=OFF && \
    cmake --build /tmp/utf8_range/build -j"$(nproc)" && \
    cmake --install /tmp/utf8_range/build && \
    rm -rf /tmp/utf8_range

# === Build and install gRPC ${GRPC_VERSION} in "package" mode for Protobuf/Abseil ===
ARG GRPC_VERSION=1.59.0
RUN git clone --depth 1 --branch v1.59.0 https://github.com/grpc/grpc.git /tmp/grpc && \
    git -C /tmp/grpc submodule update --init --recursive && \
    cmake -S /tmp/grpc -B /tmp/grpc/cmake/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/grpc" \
    -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=OFF \
    -DgRPC_PROTOBUF_PROVIDER=package \
    -DgRPC_ABSL_PROVIDER=package \
    -DgRPC_CARES_PROVIDER=module \
    -DgRPC_RE2_PROVIDER=module \
    -DgRPC_SSL_PROVIDER=module \
    -DgRPC_ZLIB_PROVIDER=module \
    -DCMAKE_PREFIX_PATH="$INSTALL_DIR/protobuf;$INSTALL_DIR/absl" && \
    cmake --build /tmp/grpc/cmake/build --target install -j"$(nproc)" && \
    rm -rf /tmp/grpc

# === Build and install StarPU ${STARPU_VERSION} ===
ARG STARPU_VERSION=1.4.8
WORKDIR /tmp/starpu
RUN wget -O /tmp/starpu.tar.gz https://gitlab.inria.fr/starpu/starpu/-/archive/starpu-${STARPU_VERSION}/starpu-starpu-${STARPU_VERSION}.tar.gz && \
    tar -xzf /tmp/starpu.tar.gz --strip-components=1 && \
    ./autogen.sh && \
    ./configure \
    --prefix="$STARPU_DIR" \
    --enable-tracing \
    --with-fxt \
    --disable-hip \
    --disable-opencl \
    --disable-mpi \
    --enable-cuda \
    --disable-fortran \
    --disable-openmp \
    && make -j"$(nproc)" && make install
WORKDIR /
RUN rm -rf /tmp/starpu /tmp/starpu.tar.gz

# Copy source code
WORKDIR /app
COPY CMakeLists.txt /app/
COPY src/ /app/src/
COPY cmake/ /app/cmake/

# Build project
WORKDIR /app/build
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} \
    -DCMAKE_PREFIX_PATH="$INSTALL_DIR/protobuf;$INSTALL_DIR/grpc;$INSTALL_DIR/utf8_range;$STARPU_DIR;$INSTALL_DIR/libtorch;$INSTALL_DIR/absl" \
    -DProtobuf_DIR=$INSTALL_DIR/protobuf/lib/cmake/protobuf \
    -DProtobuf_PROTOC_EXECUTABLE=$INSTALL_DIR/protobuf/bin/protoc \
    -DProtobuf_USE_STATIC_LIBS=ON \
    -DENABLE_COVERAGE=OFF \
    -DENABLE_SANITIZERS=OFF \
    && cmake --build . -j"$(nproc)"

# =====================
# Runtime stage
# =====================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG CUDA_VERSION
ARG UBUNTU_VERSION
ARG CMAKE_VERSION
ARG GCC_VERSION
ARG LIBTORCH_VERSION
ARG LIBTORCH_CUDA
ARG GRPC_VERSION
ARG STARPU_VERSION

LABEL maintainer="StarPU Inference Server Team" \
    version.cuda="${CUDA_VERSION}" \
    version.ubuntu="${UBUNTU_VERSION}" \
    version.cmake="${CMAKE_VERSION}" \
    version.gcc="${GCC_VERSION}" \
    version.libtorch="${LIBTORCH_VERSION}" \
    version.libtorch_cuda="${LIBTORCH_CUDA}" \
    version.grpc="${GRPC_VERSION}" \
    version.starpu="${STARPU_VERSION}"

RUN useradd -m appuser

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/appuser
ENV INSTALL_DIR=${HOME}/Install
ENV STARPU_DIR=${INSTALL_DIR}/starpu
ENV CMAKE_CUDA_ARCHITECTURES="80;86"
ENV LD_LIBRARY_PATH="$INSTALL_DIR/libtorch/lib:$INSTALL_DIR/grpc/lib:$STARPU_DIR/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

# Runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    libfxt2 \
    libhwloc15 \
    libltdl7 \
    libssl3 \
    libstdc++6 \
    && apt-get purge -y --auto-remove software-properties-common \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy artifacts from build stage
COPY --from=build --chown=appuser:appuser /root/Install/libtorch ${INSTALL_DIR}/libtorch
COPY --from=build --chown=appuser:appuser /root/Install/grpc ${INSTALL_DIR}/grpc
COPY --from=build --chown=appuser:appuser /root/Install/starpu ${STARPU_DIR}
COPY --from=build --chown=appuser:appuser /app/build/grpc_server /usr/local/bin/grpc_server
COPY --from=build --chown=appuser:appuser /app/build/grpc_client_example /usr/local/bin/grpc_client_example

RUN install -d -o appuser -g appuser /workspace
WORKDIR /workspace
USER appuser
ENTRYPOINT ["/usr/local/bin/grpc_server"]
CMD ["--help"]
