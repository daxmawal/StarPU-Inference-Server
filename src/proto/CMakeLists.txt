find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)

get_target_property(GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin LOCATION)

set(PROTO_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/grpc_service.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/model_config.proto
)

add_library(inference_proto STATIC ${PROTO_FILES})

protobuf_generate(
    TARGET inference_proto
    LANGUAGE cpp
)

protobuf_generate(
    TARGET inference_proto
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=${GRPC_CPP_PLUGIN}"
)

target_include_directories(inference_proto
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(inference_proto
    PRIVATE
        protobuf::libprotobuf
        gRPC::grpc
        gRPC::grpc++
)
