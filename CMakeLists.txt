cmake_minimum_required(VERSION 3.28)
project(starpu_server LANGUAGES CXX)

# === Options ===
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)
option(ENABLE_TSAN "Enable thread sanitizer" OFF)
option(BUILD_TESTS "Build unit tests" OFF)
option(USE_BUNDLED_DEPS "Build bundled dependencies from external/ submodules"
       ON)

if(ENABLE_SANITIZERS AND ENABLE_TSAN)
  message(
    FATAL_ERROR
      "ENABLE_SANITIZERS (ASan/UBSan) and ENABLE_TSAN cannot be enabled together."
  )
endif()

function(starpu_apply_sanitizers target_name)
  if(NOT ENABLE_SANITIZERS)
    return()
  endif()

  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(
      FATAL_ERROR
        "ENABLE_SANITIZERS requires Clang or GCC for target: ${target_name}")
  endif()

  target_compile_options(
    ${target_name} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer
                           -fno-sanitize-recover=all)
  target_link_options(${target_name} PRIVATE -fsanitize=address,undefined
                      -fno-sanitize-recover=all)
endfunction()

function(starpu_apply_tsan target_name)
  if(NOT ENABLE_TSAN)
    return()
  endif()

  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(
      FATAL_ERROR "ENABLE_TSAN requires Clang or GCC for target: ${target_name}"
    )
  endif()

  target_compile_options(${target_name} PRIVATE -fsanitize=thread
                                                -fno-omit-frame-pointer)
  target_link_options(${target_name} PRIVATE -fsanitize=thread)
endfunction()

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

  include(${PROTOBUF_ROOT_DIR}/cmake/protobuf-generate.cmake)
else()
  find_package(Protobuf REQUIRED)
  find_package(gRPC CONFIG REQUIRED)
  find_package(absl CONFIG REQUIRED)
  find_package(utf8_range CONFIG REQUIRED)
endif()

# === hwloc ===
find_package(hwloc CONFIG QUIET)
if(TARGET hwloc::hwloc AND NOT TARGET HWLOC::hwloc)
  add_library(HWLOC::hwloc ALIAS hwloc::hwloc)
endif()

if(NOT TARGET HWLOC::hwloc)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(HWLOC QUIET IMPORTED_TARGET hwloc)
  endif()

  if(TARGET PkgConfig::HWLOC)
    add_library(HWLOC::hwloc INTERFACE IMPORTED GLOBAL)
    target_link_libraries(HWLOC::hwloc INTERFACE PkgConfig::HWLOC)
    message(STATUS "hwloc found via pkg-config")
  else()
    find_path(HWLOC_INCLUDE_DIR NAMES hwloc.h REQUIRED)
    find_library(HWLOC_LIBRARY NAMES hwloc REQUIRED)
    add_library(HWLOC::hwloc UNKNOWN IMPORTED GLOBAL)
    set_target_properties(
      HWLOC::hwloc
      PROPERTIES IMPORTED_LOCATION "${HWLOC_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${HWLOC_INCLUDE_DIR}")
    message(STATUS "hwloc include directory: ${HWLOC_INCLUDE_DIR}")
    message(STATUS "hwloc library: ${HWLOC_LIBRARY}")
  endif()
else()
  message(STATUS "hwloc found via CMake package")
endif()

if(NOT DEFINED CUDAToolkit_ROOT
   AND DEFINED CUDA_TOOLKIT_ROOT_DIR
   AND EXISTS "${CUDA_TOOLKIT_ROOT_DIR}")
  set(CUDAToolkit_ROOT
      "${CUDA_TOOLKIT_ROOT_DIR}"
      CACHE PATH "CUDA toolkit root" FORCE)
elseif(
  NOT DEFINED CUDAToolkit_ROOT
  AND DEFINED CMAKE_CUDA_COMPILER
  AND EXISTS "${CMAKE_CUDA_COMPILER}")
  get_filename_component(_starpu_cuda_bin_dir "${CMAKE_CUDA_COMPILER}"
                         DIRECTORY)
  get_filename_component(_starpu_cuda_root "${_starpu_cuda_bin_dir}" DIRECTORY)
  if(EXISTS "${_starpu_cuda_root}")
    set(CUDAToolkit_ROOT
        "${_starpu_cuda_root}"
        CACHE PATH "CUDA toolkit root" FORCE)
  endif()
endif()

if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR
   AND DEFINED CUDAToolkit_ROOT
   AND EXISTS "${CUDAToolkit_ROOT}")
  set(CUDA_TOOLKIT_ROOT_DIR
      "${CUDAToolkit_ROOT}"
      CACHE PATH "CUDA toolkit root for FindCUDA compatibility" FORCE)
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
