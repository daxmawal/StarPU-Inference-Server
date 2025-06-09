cmake_minimum_required(VERSION 3.12)
project(inference_engine LANGUAGES CXX)

# === Options ===
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)

# === C++ Settings ===
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === Dependencies ===
find_package(Torch REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)

# === Subdirectories ===
add_subdirectory(src/proto)

# === StarPU ===
if (NOT DEFINED ENV{STARPU_DIR})
    message(FATAL_ERROR "STARPU_DIR environment variable must be set")
endif()

set(STARPU_DIR $ENV{STARPU_DIR})
set(STARPU_INCLUDE_DIR "${STARPU_DIR}/include")
set(STARPU_LIBRARY "${STARPU_DIR}/lib/libstarpu-1.4.so")

if(NOT EXISTS "${STARPU_INCLUDE_DIR}/starpu/1.4/starpu.h")
    message(FATAL_ERROR "STARPU_DIR does not point to a valid StarPU installation")
endif()

# === Source Files ===
set(SOURCES
    src/cli/main.cpp
    src/cli/args_parser.cpp
    src/core/inference_runner.cpp
    src/core/inference_task.cpp
    src/core/tensor_builder.cpp
    src/core/starpu_setup.cpp
    src/core/warmup.cpp
    src/utils/inference_validator.cpp
    src/utils/client_utils.cpp
    src/server/server_worker.cpp
)

# === Executable ===
add_executable(inference_engine ${SOURCES})

# === Include Paths ===
target_include_directories(inference_engine PRIVATE
    ${STARPU_INCLUDE_DIR}/starpu/1.4
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/server
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cli
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
)

# === Compiler Warnings ===
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(inference_engine PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
        -Wshadow -Wnon-virtual-dtor -Wold-style-cast
        -Wcast-align -Wunused -Woverloaded-virtual
        -Wnull-dereference -Wdouble-promotion -Wformat=2
    )
endif()

# === Coverage Flags ===
if (ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Code coverage enabled")
    set(COVERAGE_FLAGS "-fprofile-arcs -ftest-coverage -O0 -g")
    target_compile_options(inference_engine PRIVATE ${COVERAGE_FLAGS})
    target_link_options(inference_engine PRIVATE ${COVERAGE_FLAGS})
endif()

# === Sanitizers ===
if (ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Sanitizers enabled")
    set(SANITIZER_FLAGS -fsanitize=address,undefined -g)
    target_compile_options(inference_engine PRIVATE ${SANITIZER_FLAGS})
    target_link_options(inference_engine PRIVATE ${SANITIZER_FLAGS})
endif()

# === Linking ===
target_link_libraries(inference_engine PRIVATE
    ${TORCH_LIBRARIES}
    ${STARPU_LIBRARY}
    inference_proto
    gRPC::grpc++_reflection
    Threads::Threads
)

# === ABI Compatibility for libtorch ===
target_compile_definitions(inference_engine PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)

# === Status ===
message(STATUS "libtorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "StarPU include directory: ${STARPU_INCLUDE_DIR}/starpu/1.4")
message(STATUS "StarPU library: ${STARPU_LIBRARY}")