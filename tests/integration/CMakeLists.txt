add_library(integration_tests INTERFACE)

target_sources(
  integration_tests INTERFACE grpc/integration_inference_client.cpp
                              grpc/integration_inference_service.cpp)
