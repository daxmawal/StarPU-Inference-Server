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
set(HWLOC_LIBRARY "")
set(_HWLOC_CANDIDATES /usr/local/lib/libhwloc.so.15
                      /usr/lib/x86_64-linux-gnu/libhwloc.so.15)
foreach(_hwloc_path IN LISTS _HWLOC_CANDIDATES)
  if(EXISTS "${_hwloc_path}")
    set(HWLOC_LIBRARY "${_hwloc_path}")
    break()
  endif()
endforeach()
if(NOT HWLOC_LIBRARY)
  find_library(HWLOC_LIBRARY hwloc HINTS /usr/local/lib
                                         /usr/lib/x86_64-linux-gnu)
endif()
if(HWLOC_LIBRARY)
  message(STATUS "hwloc library: ${HWLOC_LIBRARY}")
endif()

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

set(_YC_BASE_DIR "${CMAKE_BINARY_DIR}/_deps")
set(_YC_SRC_DIR "${_YC_BASE_DIR}/yaml-cpp-src")
set(_YC_SUBBUILD "${_YC_BASE_DIR}/yaml-cpp-subbuild")
set(_YC_BUILD "${_YC_BASE_DIR}/yaml-cpp-build")
if(EXISTS "${_YC_SRC_DIR}" AND NOT EXISTS "${_YC_SRC_DIR}/.git")
  message(
    WARNING
      "yaml-cpp src exists without .git; purging cached _deps to force fresh Git clone."
  )
  file(REMOVE_RECURSE "${_YC_SRC_DIR}" "${_YC_SUBBUILD}" "${_YC_BUILD}")
endif()

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

# === Optional NVML support ===
find_path(NVML_INCLUDE_DIR NAMES nvml.h)
find_library(NVML_LIBRARY NAMES nvidia-ml nvml)

if(NVML_INCLUDE_DIR AND NVML_LIBRARY)
  set(STARPU_HAVE_NVML ON)
  message(STATUS "NVML found: ${NVML_LIBRARY}")
else()
  set(STARPU_HAVE_NVML OFF)
  message(STATUS "NVML not found: GPU metrics will be disabled")
endif()

# === Status Messages ===
message(STATUS "libtorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "StarPU include directory: ${STARPU_DIR}/include/starpu/1.4")
message(STATUS "StarPU library: ${STARPU_LIBRARY}")

add_subdirectory(${SRC_DIR}/grpc)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
