cmake_minimum_required(VERSION 3.28)
project(starpu_server LANGUAGES CXX)

# === Options ===
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)
option(BUILD_TESTS "Build unit tests" OFF)
option(USE_BUNDLED_DEPS "Build bundled dependencies from external/ submodules"
       ON)

# === C++ Settings ===
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# === Source Directories ===
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# === External Submodules ===
set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)

function(require_submodule path pretty_name)
  if(NOT EXISTS "${EXTERNAL_DIR}/${path}/CMakeLists.txt")
    message(
      FATAL_ERROR
        "${pretty_name} submodule missing at ${EXTERNAL_DIR}/${path}. "
        "Run `git submodule update --init --recursive` and retry.")
  endif()
endfunction()

# === Dependencies ===
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(StarPU REQUIRED)

if(USE_BUNDLED_DEPS)
  require_submodule("protobuf" "protobuf")
  # Build protobuf from the external/ submodule (no system install needed)
  set(PROTOBUF_ROOT_DIR
      ${EXTERNAL_DIR}/protobuf
      CACHE PATH "" FORCE)
  set(protobuf_INSTALL
      OFF
      CACHE BOOL "" FORCE)
  set(protobuf_BUILD_TESTS
      OFF
      CACHE BOOL "" FORCE)
  set(protobuf_BUILD_CONFORMANCE
      OFF
      CACHE BOOL "" FORCE)
  set(protobuf_BUILD_EXAMPLES
      OFF
      CACHE BOOL "" FORCE)
  set(protobuf_BUILD_SHARED_LIBS
      OFF
      CACHE BOOL "" FORCE)
  set(protobuf_BUILD_PROTOC_BINARIES
      ON
      CACHE BOOL "" FORCE)
  set(protobuf_BUILD_PROTOBUF_BINARIES
      ON
      CACHE BOOL "" FORCE)
  set(protobuf_WITH_ZLIB
      OFF
      CACHE BOOL "" FORCE)

  # gRPC with bundled third_party deps (absl/protobuf/re2/cares/boringssl/zlib)
  require_submodule("grpc" "gRPC")
  set(gRPC_PROTOBUF_PROVIDER
      module
      CACHE STRING "" FORCE)
  set(gRPC_INSTALL
      OFF
      CACHE BOOL "" FORCE)
  set(gRPC_BUILD_TESTS
      OFF
      CACHE BOOL "" FORCE)
  set(gRPC_BUILD_CODEGEN
      ON
      CACHE BOOL "" FORCE)
  set(gRPC_BUILD_GRPC_CPP_PLUGIN
      ON
      CACHE BOOL "" FORCE)
  set(gRPC_BUILD_GRPC_CSHARP_PLUGIN
      OFF
      CACHE BOOL "" FORCE)
  set(gRPC_BUILD_GRPC_NODE_PLUGIN
      OFF
      CACHE BOOL "" FORCE)
  set(gRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN
      OFF
      CACHE BOOL "" FORCE)
  set(gRPC_BUILD_GRPC_PHP_PLUGIN
      OFF
      CACHE BOOL "" FORCE)
  set(gRPC_BUILD_GRPC_PYTHON_PLUGIN
      OFF
      CACHE BOOL "" FORCE)
  set(gRPC_BUILD_GRPC_RUBY_PLUGIN
      OFF
      CACHE BOOL "" FORCE)
  set(gRPC_ABSL_PROVIDER
      module
      CACHE STRING "" FORCE)
  set(gRPC_RE2_PROVIDER
      module
      CACHE STRING "" FORCE)
  set(gRPC_CARES_PROVIDER
      module
      CACHE STRING "" FORCE)
  set(gRPC_SSL_PROVIDER
      module
      CACHE STRING "" FORCE)
  set(gRPC_ZLIB_PROVIDER
      module
      CACHE STRING "" FORCE)
  add_subdirectory(${EXTERNAL_DIR}/grpc)

  # Ensure gRPC namespace aliases exist (some gRPC builds skip alias
  # definitions)
  if(TARGET grpc AND NOT TARGET gRPC::grpc)
    add_library(gRPC::grpc ALIAS grpc)
  endif()
  if(TARGET grpc++ AND NOT TARGET gRPC::grpc++)
    add_library(gRPC::grpc++ ALIAS grpc++)
  endif()
  if(TARGET grpc++_reflection AND NOT TARGET gRPC::grpc++_reflection)
    add_library(gRPC::grpc++_reflection ALIAS grpc++_reflection)
  endif()
  if(TARGET grpc_cpp_plugin AND NOT TARGET gRPC::grpc_cpp_plugin)
    add_executable(gRPC::grpc_cpp_plugin ALIAS grpc_cpp_plugin)
  endif()
  # utf8_range from gRPC's third_party
  if(TARGET utf8_range AND NOT TARGET utf8_range::utf8_range)
    add_library(utf8_range::utf8_range ALIAS utf8_range)
  endif()
  if(TARGET utf8_validity AND NOT TARGET utf8_range::utf8_validity)
    add_library(utf8_range::utf8_validity ALIAS utf8_validity)
  endif()

  set(ABSL_LIBS absl::log absl::strings absl::base absl::log_internal_check_op
                absl::log_internal_message absl::raw_logging_internal)
  include(${PROTOBUF_ROOT_DIR}/cmake/protobuf-generate.cmake)
  set(Protobuf_LIBRARIES protobuf::libprotobuf)
  set(Protobuf_INCLUDE_DIRS
      ${PROTOBUF_ROOT_DIR}/src
      ${CMAKE_BINARY_DIR}/external/grpc/third_party/protobuf)
  set(Protobuf_PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
else()
  find_package(Protobuf REQUIRED)
  find_package(gRPC CONFIG REQUIRED)
  find_package(absl CONFIG REQUIRED)
  find_package(utf8_range CONFIG REQUIRED)

  set(ABSL_LIBS absl::log absl::strings absl::base absl::log_internal_check_op)
endif()
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

if(BUILD_TESTS)
  enable_testing()
  set(GTEST_DIR "${EXTERNAL_DIR}/googletest")
  require_submodule("googletest" "googletest")
  add_subdirectory(${GTEST_DIR})
endif()

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
require_submodule("prometheus-cpp" "prometheus-cpp")
add_subdirectory(${EXTERNAL_DIR}/prometheus-cpp)

add_subdirectory(${SRC_DIR}/proto)

# === yaml-cpp via submodule ===
set(YAML_CPP_BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS
    OFF
    CACHE BOOL "" FORCE)
require_submodule("yaml-cpp" "yaml-cpp")
add_subdirectory(${EXTERNAL_DIR}/yaml-cpp)

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
include(FindTorchCustom)
message(STATUS "libtorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "StarPU include directory: ${STARPU_DIR}/include/starpu/1.4")
message(STATUS "StarPU library: ${STARPU_LIBRARY}")

add_subdirectory(${SRC_DIR}/grpc)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
