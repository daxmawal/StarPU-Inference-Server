cmake_minimum_required(VERSION 3.12)
project(inference_engine)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# External Dependencies
set(CMAKE_PREFIX_PATH "/local/home/jd258565/Install/libtorch")
find_package(Torch REQUIRED)

# StarPU Configuration
set(STARPU_DIR $ENV{STARPU_DIR})
set(STARPU_INCLUDE_DIR "${STARPU_DIR}/include/starpu/1.4")
set(STARPU_LIBRARY "${STARPU_DIR}/lib/libstarpu-1.4.so")

# Include Directories
include_directories(
    ${STARPU_INCLUDE_DIR}
    src
)

# Executable Definition
add_executable(inference_engine
    src/main.cpp
    src/args_parser.cpp
    src/inference_runner.cpp
    src/inference_task.cpp
    src/inference_validator.cpp
)

# Target Includes
target_include_directories(inference_engine PRIVATE
    src
    ${STARPU_INCLUDE_DIR}
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
message(STATUS "StarPU include directory: ${STARPU_INCLUDE_DIR}")
message(STATUS "StarPU library: ${STARPU_LIBRARY}")