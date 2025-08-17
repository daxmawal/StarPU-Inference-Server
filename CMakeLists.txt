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
find_package(utf8_range CONFIG REQUIRED)

set(ABSL_LIBS absl::log absl::strings absl::base)

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
set(DEBUG_MIN_FRAME_SIZE
    1024
    CACHE STRING "Minimum stack frame size for warning")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(
    starpu_server
    PRIVATE -Wall
            -Wextra
            -Wpedantic
            -Walloca
            -Wcast-align
            -Wcast-qual
            -Wcomma-subscript
            -Wctor-dtor-privacy
            -Wdeprecated-copy-dtor
            -Wdouble-promotion
            -Wduplicated-branches
            -Wduplicated-cond
            -Wenum-conversion
            -Wextra-semi
            -Wfloat-equal
            -Wformat=2
            -Wframe-larger-than=${DEBUG_MIN_FRAME_SIZE}
            -Wlogical-op
            -Wmismatched-tags
            -Wmissing-braces
            -Wmultichar
            -Wnoexcept
            -Wnon-virtual-dtor
            -Woverloaded-virtual
            -Wpointer-arith
            -Wrange-loop-construct
            -Wrestrict
            -Wshadow
            -Wstrict-null-sentinel
            -Wsuggest-attribute=format
            -Wsuggest-attribute=malloc
            -Wvla
            -Wvolatile
            -Wwrite-strings
            -Wmissing-declarations
            -Wmissing-include-dirs
            -Winline
            -Wundef
            -Wnull-dereference
            -Wold-style-cast)
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

if(ENABLE_COVERAGE AND ENABLE_SANITIZERS)
  message(FATAL_ERROR "ENABLE_COVERAGE and ENABLE_SANITIZERS are incompatible.")
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
