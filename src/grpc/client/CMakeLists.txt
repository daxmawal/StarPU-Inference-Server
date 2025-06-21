add_executable(grpc_client_example client_main.cpp client_args.cpp)
set_target_properties(grpc_client_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_compile_features(grpc_client_example PRIVATE cxx_std_20)
target_include_directories(grpc_client_example PRIVATE ${CMAKE_SOURCE_DIR}/src ${STARPU_DIR}/include/starpu/1.4)
target_compile_definitions(grpc_client_example PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)
target_link_libraries(grpc_client_example PRIVATE
    ${TORCH_LIBRARIES}
    inference_proto
    gRPC::grpc++_reflection
    Threads::Threads
    ${Protobuf_LIBRARIES}
    ${ABSL_LIBS}
)