set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(UTILS_DIR ${SRC_DIR}/utils)

add_executable(
  grpc_client_example client_main.cpp client_args.cpp inference_client.cpp
                      ${UTILS_DIR}/time_utils.cpp)

set_target_properties(grpc_client_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                     ${CMAKE_BINARY_DIR})

target_compile_features(grpc_client_example PRIVATE cxx_std_23)
target_compile_definitions(grpc_client_example PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_include_directories(grpc_client_example
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(
  grpc_client_example
  PRIVATE Torch::Torch
          inference_proto
          gRPC::grpc++_reflection
          Threads::Threads
          ${Protobuf_LIBRARIES}
          utf8_range::utf8_range
          utf8_range::utf8_validity
          absl::log
          absl::strings
          absl::base)
