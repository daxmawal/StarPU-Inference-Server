cmake_minimum_required(VERSION 3.12)
project(inference_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/local/home/jd258565/Install/libtorch")

find_package(Torch REQUIRED)

set(STARPU_DIR $ENV{STARPU_DIR})
set(STARPU_INCLUDE_DIR "${STARPU_DIR}/include/starpu/1.4")
set(STARPU_LIBRARY "${STARPU_DIR}/lib/libstarpu-1.4.so")

include_directories(${STARPU_INCLUDE_DIR})
add_executable(inference_engine src/main.cpp)

target_include_directories(inference_engine PRIVATE ${STARPU_INCLUDE_DIR})
target_link_libraries(inference_engine "${TORCH_LIBRARIES}" ${STARPU_LIBRARY})
target_compile_definitions(inference_engine PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)


message(STATUS "libtorch: ${TORCH_LIBRARIES}")
message(STATUS "StarPU include: ${STARPU_INCLUDE_DIR}")
message(STATUS "StarPU lib: ${STARPU_LIBRARY}")
