add_library(unit_tests INTERFACE)

target_sources(
  unit_tests
  INTERFACE cli/cli_tests.cpp
            core/test_inference_runner.cpp
            core/test_inference_runner_helpers.cpp
            core/test_tensor_builder.cpp
            core/test_inference_task.cpp
            core/test_inference_task_errors.cpp
            core/test_inference_task_limits.cpp
            core/test_inference_queue.cpp
            core/test_run_inference.cpp
            core/test_warmup_runner.cpp
            grpc/unit_inference_service.cpp
            grpc/unit_client.cpp
            grpc/robustness/edgecases_client.cpp
            grpc/robustness/edgecases_inference_service.cpp
            integration/test_run_inference_loop.cpp
            integration/test_grpc_client_server.cpp
            starpu/unit_starpu_setup.cpp
            starpu/robustness/edgecases_starpu_setup.cpp
            starpu/test_starpu_task_runner.cpp
            starpu/test_tensor_builder_from_starpu.cpp
            utils/test_client_utils.cpp
            utils/test_datatype_utils.cpp
            utils/test_logger.cpp
            utils/test_transparent_hash.cpp
            utils/test_input_generator.cpp
            utils/test_inference_validator.cpp)
