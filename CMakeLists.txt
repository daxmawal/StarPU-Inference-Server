cmake_minimum_required(VERSION 3.12)
project(inference_engine)

option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# External Dependencies
set(CMAKE_PREFIX_PATH "/local/home/jd258565/Install/libtorch")
find_package(Torch REQUIRED)

# StarPU Configuration
set(STARPU_DIR $ENV{STARPU_DIR})
set(STARPU_INCLUDE_DIR "${STARPU_DIR}/include")
set(STARPU_LIBRARY "${STARPU_DIR}/lib/libstarpu-1.4.so")

# Include Directories
include_directories(
    ${STARPU_INCLUDE_DIR}/starpu/1.4
    src
)

if (ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Code coverage enabled")
    set(COVERAGE_FLAGS "-fprofile-arcs -ftest-coverage -O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COVERAGE_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${COVERAGE_FLAGS}")
endif()

# Executable Definition
add_executable(inference_engine
    src/cli/main.cpp
    src/cli/args_parser.cpp
    src/core/inference_runner.cpp
    src/core/inference_task.cpp
    src/utils/inference_validator.cpp
    src/core/tensor_builder.cpp
    src/core/starpu_setup.cpp
    src/core/warmup.cpp
    src/server/server_worker.cpp
    src/utils/client_utils.cpp
)

if (ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Sanitizers enabled")
    set(SANITIZER_FLAGS -fsanitize=address,undefined -g)
    target_compile_options(inference_engine PRIVATE ${SANITIZER_FLAGS})
    target_link_options(inference_engine PRIVATE ${SANITIZER_FLAGS})
endif()

# Target Includes
target_include_directories(inference_engine PRIVATE
    ${STARPU_INCLUDE_DIR}/starpu/1.4
    src
    src/core
    src/server
    src/cli
    src/utils
)

# Link Libraries
target_link_libraries(inference_engine PRIVATE
    ${TORCH_LIBRARIES}
    ${STARPU_LIBRARY}
)

# Compile Definitions
target_compile_definitions(inference_engine PRIVATE
    -D_GLIBCXX_USE_CXX11_ABI=1
)

# Compiler Options
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(inference_engine PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
    )
endif()

# Messages for verification
message(STATUS "libtorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "StarPU include directory: ${STARPU_INCLUDE_DIR}/starpu/1.4")
message(STATUS "StarPU library: ${STARPU_LIBRARY}")