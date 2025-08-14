add_library(integration_tests INTERFACE)

target_sources(
  integration_tests
  INTERFACE grpc/integration_inference_client.cpp
            grpc/integration_inference_service.cpp
            starpu/integration_starpu_setup.cpp
            starpu/integration_starpu_task_runner.cpp
            starpu/integration_tensor_builder_from_starpu.cpp
            utils/integration_client_utils.cpp
            utils/integration_datatype_utils.cpp
            utils/integration_inference_validator.cpp
            utils/integration_input_generator.cpp
            utils/integration_logger.cpp
            utils/integration_transparent_hash.cpp
            cli/integration_cli.cpp)
