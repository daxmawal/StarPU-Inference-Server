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
find_package(utf8_range CONFIG REQUIRED)

# Optional: NVTX instrumentation (nsys --trace=nvtx)
set(HAVE_NVTX OFF)
find_package(CUDAToolkit QUIET)
include(CheckIncludeFileCXX)
if(CUDAToolkit_FOUND)
  set(CMAKE_REQUIRED_INCLUDES "${CUDAToolkit_INCLUDE_DIRS}")
endif()
check_include_file_cxx("nvtx3/nvToolsExt.h" HAVE_NVTX3_HEADER)
unset(CMAKE_REQUIRED_INCLUDES)

if(HAVE_NVTX3_HEADER)
  set(HAVE_NVTX ON)
endif()

set(ABSL_LIBS absl::log absl::strings absl::base absl::log_internal_check_op)

include(FetchContent)
if(BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip)
  FetchContent_MakeAvailable(googletest)
endif()

set(_PC_BASE_DIR "${CMAKE_BINARY_DIR}/_deps")
set(_PC_SRC_DIR "${_PC_BASE_DIR}/prometheus-cpp-src")
set(_PC_SUBBUILD "${_PC_BASE_DIR}/prometheus-cpp-subbuild")
set(_PC_BUILD "${_PC_BASE_DIR}/prometheus-cpp-build")
if(EXISTS "${_PC_SRC_DIR}" AND NOT EXISTS "${_PC_SRC_DIR}/.git")
  message(
    WARNING
      "prometheus-cpp src exists without .git; purging cached _deps to force fresh Git clone."
  )
  file(REMOVE_RECURSE "${_PC_SRC_DIR}" "${_PC_SUBBUILD}" "${_PC_BUILD}")
endif()

FetchContent_Declare(
  prometheus-cpp
  GIT_REPOSITORY https://github.com/jupp0r/prometheus-cpp.git
  GIT_TAG v1.1.0
  GIT_SHALLOW TRUE
  GIT_SUBMODULES_RECURSE TRUE)
set(ENABLE_PULL
    ON
    CACHE BOOL "" FORCE)
set(ENABLE_PUSH
    OFF
    CACHE BOOL "" FORCE)
set(ENABLE_COMPRESSION
    OFF
    CACHE BOOL "" FORCE)
set(ENABLE_TESTING
    OFF
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(prometheus-cpp)

add_subdirectory(${SRC_DIR}/proto)

# === yaml-cpp via FetchContent ===
include(FetchContent)

set(YAML_CPP_BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS
    OFF
    CACHE BOOL "" FORCE)

FetchContent_Declare(yaml-cpp
                     GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git)

FetchContent_MakeAvailable(yaml-cpp)

# Suppress noisy Clang warning from yaml-cpp's dragonbox header Seen as:
# [-Wconstexpr-not-const] while yaml-cpp builds with older C++ modes
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(TARGET yaml-cpp)
    target_compile_options(yaml-cpp PRIVATE -Wno-constexpr-not-const)
  endif()
endif()

# === Executable ===
add_executable(starpu_server)
target_sources(
  starpu_server
  PRIVATE ${SRC_DIR}/cli/args_parser.cpp
          ${SRC_DIR}/cli/main.cpp
          ${SRC_DIR}/core/inference_runner.cpp
          ${SRC_DIR}/core/inference_task.cpp
          ${SRC_DIR}/core/input_slot_pool.cpp
          ${SRC_DIR}/core/output_slot_pool.cpp
          ${SRC_DIR}/core/starpu_setup.cpp
          ${SRC_DIR}/core/tensor_builder.cpp
          ${SRC_DIR}/core/warmup.cpp
          ${SRC_DIR}/utils/client_utils.cpp
          ${SRC_DIR}/utils/config_loader.cpp
          ${SRC_DIR}/utils/inference_validator.cpp
          ${SRC_DIR}/utils/perf_observer.cpp
          ${SRC_DIR}/utils/time_utils.cpp
          ${SRC_DIR}/monitoring/metrics.cpp
          ${SRC_DIR}/starpu_task_worker/starpu_task_worker.cpp)

target_compile_features(starpu_server PRIVATE cxx_std_23)
target_compile_definitions(starpu_server PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

# === Include Directories ===
target_include_directories(
  starpu_server PRIVATE ${SRC_DIR} ${SRC_DIR}/core ${SRC_DIR}/utils
                        ${SRC_DIR}/cli ${SRC_DIR}/starpu_task_worker)

find_path(NVML_INCLUDE_DIR NAMES nvml.h)
find_library(NVML_LIBRARY NAMES nvidia-ml nvml)

if(NVML_INCLUDE_DIR AND NVML_LIBRARY)
  message(STATUS "NVML found: ${NVML_LIBRARY}")
  target_compile_definitions(starpu_server PRIVATE STARPU_HAVE_NVML=1)
  target_include_directories(starpu_server PRIVATE ${NVML_INCLUDE_DIR})
  target_link_libraries(starpu_server PRIVATE ${NVML_LIBRARY})
else()
  message(STATUS "NVML not found: GPU metrics will be disabled")
endif()

# === Compiler Warnings ===
set(DEBUG_MIN_FRAME_SIZE
    8192
    CACHE STRING "Minimum stack frame size for warning")

# Fine-grained warnings by compiler to avoid unknown-warning-option noise
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
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
            -Wundef
            -Wold-style-cast)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(
    starpu_server
    PRIVATE -Wall
            -Wextra
            -Wpedantic
            -Walloca
            -Wcast-align
            -Wcast-qual
            -Wdouble-promotion
            -Wenum-conversion
            -Wextra-semi
            -Wfloat-equal
            -Wformat=2
            -Wframe-larger-than=${DEBUG_MIN_FRAME_SIZE}
            -Wmismatched-tags
            -Wmissing-braces
            -Wmultichar
            -Wnon-virtual-dtor
            -Woverloaded-virtual
            -Wpointer-arith
            -Wrange-loop-construct
            -Wshadow
            -Wvla
            -Wwrite-strings
            -Wmissing-declarations
            -Wmissing-include-dirs
            -Wundef
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
  starpu_server
  PRIVATE Torch::Torch
          protobuf::libprotobuf
          gRPC::grpc++
          utf8_range::utf8_range
          StarPU::starpu
          yaml-cpp::yaml-cpp
          prometheus-cpp::pull
          ${ABSL_LIBS})

if(HAVE_NVTX)
  message(STATUS "NVTX 3 header found: enabling NVTX instrumentation")
  target_compile_definitions(starpu_server PRIVATE HAVE_NVTX=1)
  if(CUDAToolkit_FOUND)
    target_include_directories(starpu_server
                               PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  endif()
endif()

# === Status Messages ===
message(STATUS "libtorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "StarPU include directory: ${STARPU_DIR}/include/starpu/1.4")
message(STATUS "StarPU library: ${STARPU_LIBRARY}")

add_subdirectory(${SRC_DIR}/grpc)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
