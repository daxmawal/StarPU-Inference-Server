set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(UTILS_DIR ${SRC_DIR}/utils)

add_executable(client_example client_main.cpp client_args.cpp
                              inference_client.cpp ${UTILS_DIR}/time_utils.cpp)

set_target_properties(client_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                ${CMAKE_BINARY_DIR})

target_compile_features(client_example PRIVATE cxx_std_23)
target_compile_definitions(client_example PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_include_directories(client_example PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(client_example PRIVATE Torch::Torch inference_proto
                                             Threads::Threads gRPC::grpc++)

if(TARGET gRPC::grpc++_reflection)
  target_link_libraries(client_example PRIVATE gRPC::grpc++_reflection)
elseif(TARGET grpc++_reflection)
  target_link_libraries(client_example PRIVATE grpc++_reflection)
else()
  message(
    WARNING
      "gRPC reflection library not found; client_example will be built without reflection."
  )
endif()
