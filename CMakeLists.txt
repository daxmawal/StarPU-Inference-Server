cmake_minimum_required(VERSION 3.12)
project(starpu_server LANGUAGES CXX)

# === Options ===
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)

# === C++ Settings ===
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# === Source Directories ===
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CLI_DIR ${SRC_DIR}/cli)
set(CORE_DIR ${SRC_DIR}/core)
set(UTILS_DIR ${SRC_DIR}/utils)
set(STARPU_TASK_WORKER_DIR ${SRC_DIR}/starpu_task_worker)

# === Dependencies ===
find_package(Torch REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)

add_subdirectory(${SRC_DIR}/proto)

# === StarPU ===
set(STARPU_DIR $ENV{STARPU_DIR})
if (NOT STARPU_DIR OR NOT EXISTS "${STARPU_DIR}/include/starpu/1.4/starpu.h")
    message(FATAL_ERROR "Valid STARPU_DIR environment variable must be set")
endif()

message(STATUS "Using STARPU_DIR: ${STARPU_DIR}")
include_directories(${STARPU_DIR}/include)
link_directories(${STARPU_DIR}/lib)

# === Abseil (gRPC dependency) ===
set(ABSL_HINT "$ENV{HOME}/Install/grpc/lib")
foreach(lib IN ITEMS absl_log_internal_check_op absl_strings absl_base)
    find_library(${lib}_PATH ${lib} HINTS "${ABSL_HINT}" REQUIRED)
    if(NOT ${lib}_PATH)
        message(FATAL_ERROR "Could not find ${lib} in ${ABSL_HINT}")
    endif()
    list(APPEND ABSL_LIBS ${${lib}_PATH})
endforeach()

# === Source Files ===
file(GLOB_RECURSE SOURCES
    ${CLI_DIR}/*.cpp
    ${CORE_DIR}/*.cpp
    ${UTILS_DIR}/*.cpp
    ${STARPU_TASK_WORKER_DIR}/*.cpp
)

# === Executable ===
add_executable(starpu_server ${SOURCES})

# === C++ Features ===
target_compile_features(starpu_server PRIVATE cxx_std_23)

# === Include Paths ===
target_include_directories(starpu_server PRIVATE
    ${STARPU_DIR}/include/starpu/1.4
    ${SRC_DIR}
    ${CORE_DIR}
    ${UTILS_DIR}
    ${CLI_DIR}
    ${STARPU_TASK_WORKER_DIR}
)

# === Compiler Warnings ===
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(starpu_server PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
        -Wshadow -Wold-style-cast -Wcast-align -Wunused
        -Woverloaded-virtual -Wnull-dereference -Wdouble-promotion -Wformat=2
    )
endif()

# === Coverage Flags ===
if (ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(starpu_server PRIVATE -fprofile-arcs -ftest-coverage -O0 -g)
    target_link_options(starpu_server PRIVATE -fprofile-arcs -ftest-coverage -O0 -g)
endif()

# === Sanitizers ===
if (ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(starpu_server PRIVATE -fsanitize=address,undefined -g)
    target_link_options(starpu_server PRIVATE -fsanitize=address,undefined -g)
endif()

# === Linking ===
target_link_libraries(starpu_server PRIVATE
    ${TORCH_LIBRARIES}
    starpu-1.4
)

# === Compatibility for libtorch ===
target_compile_definitions(starpu_server PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)

# === Status Messages ===
message(STATUS "libtorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "StarPU include directory: ${STARPU_DIR}/include/starpu/1.4")
message(STATUS "StarPU library: ${STARPU_DIR}/lib/libstarpu-1.4.so")

add_subdirectory(${SRC_DIR}/grpc)