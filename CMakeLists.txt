cmake_minimum_required(VERSION 3.28)
project(starpu_server LANGUAGES CXX)

# === Options ===
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)
option(BUILD_TESTS "Build unit tests" OFF)

# === C++ Settings ===
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# === Source Directories ===
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# === Dependencies ===
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindTorchCustom)
find_package(StarPU REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(Threads REQUIRED)

set(ABSL_LIBS absl::log_internal_check_op absl::strings absl::base)

include(FetchContent)
if(BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip)
  FetchContent_MakeAvailable(googletest)
endif()

add_subdirectory(${SRC_DIR}/proto)

# === Executable ===
add_executable(starpu_server)
target_sources(
  starpu_server
  PRIVATE ${SRC_DIR}/cli/args_parser.cpp
          ${SRC_DIR}/cli/main.cpp
          ${SRC_DIR}/core/inference_runner.cpp
          ${SRC_DIR}/core/inference_task.cpp
          ${SRC_DIR}/core/starpu_setup.cpp
          ${SRC_DIR}/core/tensor_builder.cpp
          ${SRC_DIR}/core/warmup.cpp
          ${SRC_DIR}/utils/client_utils.cpp
          ${SRC_DIR}/utils/inference_validator.cpp
          ${SRC_DIR}/utils/time_utils.cpp
          ${SRC_DIR}/starpu_task_worker/starpu_task_worker.cpp)

target_compile_features(starpu_server PRIVATE cxx_std_23)
target_compile_definitions(starpu_server PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

# === Include Directories ===
target_include_directories(
  starpu_server PRIVATE ${SRC_DIR} ${SRC_DIR}/core ${SRC_DIR}/utils
                        ${SRC_DIR}/cli ${SRC_DIR}/starpu_task_worker)

# === Compiler Warnings ===
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(
    starpu_server
    PRIVATE -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
            -Wshadow
            -Wold-style-cast
            -Wcast-align
            -Wunused
            -Woverloaded-virtual
            -Wnull-dereference
            -Wdouble-promotion
            -Wpadded
            -Wformat=2
            -Wundef
            -Wmissing-declarations
            -Wmissing-include-dirs
            -Winline)
endif()

# === Coverage Flags ===
if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(starpu_server PRIVATE -fprofile-arcs -ftest-coverage
                                               -O0 -g)
  target_link_options(starpu_server PRIVATE -fprofile-arcs -ftest-coverage -O0
                      -g)
endif()

# === Sanitizers ===
if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(starpu_server PRIVATE -fsanitize=address,undefined -g)
  target_link_options(starpu_server PRIVATE -fsanitize=address,undefined -g)
endif()

# === Linking ===
target_link_libraries(
  starpu_server PRIVATE Torch::Torch protobuf::libprotobuf gRPC::grpc++
                        StarPU::starpu ${ABSL_LIBS})

# === Status Messages ===
message(STATUS "libtorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "StarPU include directory: ${STARPU_DIR}/include/starpu/1.4")
message(STATUS "StarPU library: ${STARPU_LIBRARY}")

add_subdirectory(${SRC_DIR}/grpc)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
