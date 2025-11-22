ARG CUDA_VERSION=11.8.0
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/root
ENV INSTALL_DIR=${HOME}/Install
ENV STARPU_DIR=${INSTALL_DIR}/starpu
ENV LD_LIBRARY_PATH="$INSTALL_DIR/libtorch/lib:$STARPU_DIR/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# === Install CMake ${CMAKE_VERSION} ===
ARG CMAKE_VERSION=3.28.3
ADD https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz /tmp/cmake-linux-x86_64.tar.gz
RUN tar --strip-components=1 -xz -f /tmp/cmake-linux-x86_64.tar.gz -C /usr/local && \
    rm /tmp/cmake-linux-x86_64.tar.gz

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
ADD https://download.pytorch.org/libtorch/${LIBTORCH_CUDA}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_CUDA}.zip /tmp/libtorch.zip
RUN mkdir -p "$INSTALL_DIR/libtorch" && \
    unzip /tmp/libtorch.zip -d "$INSTALL_DIR" && \
    rm /tmp/libtorch.zip

# === Build and install StarPU ${STARPU_VERSION} ===
ARG STARPU_VERSION=1.4.8
WORKDIR /tmp/starpu
ADD https://gitlab.inria.fr/starpu/starpu/-/archive/starpu-${STARPU_VERSION}/starpu-starpu-${STARPU_VERSION}.tar.gz /tmp/starpu.tar.gz
RUN tar -xzf /tmp/starpu.tar.gz --strip-components=1 && \
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

# Copy only the build inputs
WORKDIR /app
COPY CMakeLists.txt ./
COPY cmake ./cmake
COPY external ./external
COPY src ./src
COPY tests ./tests

# Build project
WORKDIR /app/build
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DUSE_BUNDLED_DEPS=ON \
    -DCMAKE_PREFIX_PATH="$STARPU_DIR;$INSTALL_DIR/libtorch" \
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
ENV LD_LIBRARY_PATH="$INSTALL_DIR/libtorch/lib:$STARPU_DIR/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

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
COPY --from=build --chown=appuser:appuser /root/Install/starpu ${STARPU_DIR}
COPY --from=build --chown=appuser:appuser /app/build/starpu_server /usr/local/bin/starpu_server
COPY --from=build --chown=appuser:appuser /app/build/client_example /usr/local/bin/client_example

RUN install -d -o appuser -g appuser /workspace
WORKDIR /workspace
USER appuser
ENTRYPOINT ["/usr/local/bin/starpu_server"]
CMD ["--help"]
