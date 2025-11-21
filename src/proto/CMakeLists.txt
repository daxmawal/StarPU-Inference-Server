if(NOT TARGET protobuf::libprotobuf)
  find_package(Protobuf REQUIRED)
endif()
if(NOT TARGET gRPC::grpc++)
  find_package(gRPC CONFIG REQUIRED)
endif()

if(TARGET gRPC::grpc_cpp_plugin)
  set(GRPC_CPP_PLUGIN "$<TARGET_FILE:gRPC::grpc_cpp_plugin>")
elseif(TARGET grpc_cpp_plugin)
  set(GRPC_CPP_PLUGIN "$<TARGET_FILE:grpc_cpp_plugin>")
else()
  find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
  if(NOT GRPC_CPP_PLUGIN)
    message(
      FATAL_ERROR
        "grpc_cpp_plugin not found. Install gRPC C++ plugin or build gRPC with codegen enabled."
    )
  endif()
endif()

set(PROTO_FILES ${CMAKE_CURRENT_SOURCE_DIR}/grpc_service.proto
                ${CMAKE_CURRENT_SOURCE_DIR}/model_config.proto)

add_library(inference_proto STATIC ${PROTO_FILES})

protobuf_generate(TARGET inference_proto LANGUAGE cpp)

protobuf_generate(
  TARGET
  inference_proto
  LANGUAGE
  grpc
  GENERATE_EXTENSIONS
  .grpc.pb.h
  .grpc.pb.cc
  PLUGIN
  "protoc-gen-grpc=${GRPC_CPP_PLUGIN}")

target_include_directories(inference_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
                                                  ${Protobuf_INCLUDE_DIRS})

target_link_libraries(inference_proto PUBLIC protobuf::libprotobuf gRPC::grpc
                                             gRPC::grpc++ ${ABSL_LIBS})
