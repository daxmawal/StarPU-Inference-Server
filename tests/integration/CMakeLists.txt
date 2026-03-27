add_library(
  integration_core_tests_obj OBJECT
  core/integration_inference_queue.cpp
  core/integration_inference_runner_helpers.cpp
  core/integration_inference_task_errors.cpp
  core/integration_warmup_runner.cpp
  core/integration_inference_runner.cpp
  core/integration_inference_task.cpp
  core/integration_inference_task_limits.cpp
  core/integration_tensor_builder.cpp)

add_library(
  integration_grpc_tests_obj OBJECT
  grpc/integration_inference_client.cpp grpc/integration_inference_service.cpp
  grpc/integration_grpc_client_server.cpp)

add_library(
  integration_starpu_tests_obj OBJECT
  starpu/integration_starpu_setup.cpp
  starpu/integration_tensor_builder_from_starpu.cpp)

add_library(integration_utils_tests_obj OBJECT
            utils/integration_skipped_modules.cpp)
