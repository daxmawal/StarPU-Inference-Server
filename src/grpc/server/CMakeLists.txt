file(GLOB_RECURSE SERVER_SOURCES
    ${CLI_DIR}/args_parser.cpp
    ${CORE_DIR}/*.cpp
    ${UTILS_DIR}/*.cpp
    ${STARPU_TASK_WORKER_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_executable(grpc_server ${SERVER_SOURCES})

set_target_properties(grpc_server PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_compile_features(grpc_server PRIVATE cxx_std_23)
target_include_directories(grpc_server PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CLI_DIR}
    ${CORE_DIR}
    ${UTILS_DIR}
    ${STARPU_TASK_WORKER_DIR}
    ${STARPU_DIR}/include/starpu/1.4
)
target_compile_definitions(grpc_server PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)
target_link_libraries(grpc_server PRIVATE
    ${TORCH_LIBRARIES}
    inference_proto
    gRPC::grpc++_reflection
    Threads::Threads
    ${Protobuf_LIBRARIES}
    ${ABSL_LIBS}
    ${STARPU_DIR}/lib/libstarpu-1.4.so 
)