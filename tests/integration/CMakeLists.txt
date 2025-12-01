add_library(integration_tests INTERFACE)

target_sources(
  integration_tests
  INTERFACE grpc/integration_inference_client.cpp
            grpc/integration_inference_service.cpp
            grpc/integration_grpc_client_server.cpp
            starpu/integration_starpu_setup.cpp
            starpu/integration_starpu_task_runner.cpp
            starpu/integration_tensor_builder_from_starpu.cpp
            utils/integration_skipped_modules.cpp
            utils/integration_inference_validator.cpp
            core/integration_inference_queue.cpp
            core/integration_inference_runner_helpers.cpp
            core/integration_inference_task_errors.cpp
            core/integration_warmup_runner.cpp
            core/integration_inference_runner.cpp
            core/integration_inference_task.cpp
            core/integration_inference_task_limits.cpp
            core/integration_tensor_builder.cpp)
