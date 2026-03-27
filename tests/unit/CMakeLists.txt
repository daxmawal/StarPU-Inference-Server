add_library(
  unit_core_tests_obj OBJECT
  core/unit_inference_runner_helpers.cpp
  core/unit_inference_task_errors.cpp
  core/unit_inference_runner.cpp
  core/unit_synthesize_outputs.cpp
  core/unit_latency_statistics.cpp
  core/unit_inference_task.cpp
  core/unit_run_with_logged_exceptions.cpp
  core/unit_tensor_builder.cpp
  core/unit_warmup_runner.cpp
  core/unit_slot_pool_base.cpp
  core/unit_starpu_setup.cpp
  core/unit_input_slot_pool_deleter.cpp
  core/robustness/edgecases_inference_queue.cpp
  core/robustness/edgecases_warmup_runner.cpp
  core/robustness/edgecases_inference_runner.cpp
  core/robustness/edgecases_inference_task.cpp
  core/robustness/edgecases_tensor_builder.cpp)

add_library(
  unit_grpc_tests_obj OBJECT
  grpc/async_server_context_test.cpp
  grpc/resolve_model_name_test.cpp
  grpc/server_main_test.cpp
  grpc/unary_call_data_guard_test.cpp
  grpc/unit_inference_service.cpp
  grpc/unit_client.cpp
  grpc/inference_client_helpers_test.cpp
  grpc/robustness/edgecases_client.cpp
  grpc/robustness/edgecases_inference_service.cpp)

add_library(
  unit_starpu_tests_obj OBJECT
  starpu/unit_starpu_task_runner_support.cpp
  starpu/unit_starpu_task_runner_callbacks_and_tracing.cpp
  starpu/unit_starpu_task_runner_submission_and_slots.cpp
  starpu/unit_starpu_task_runner_batching_and_internal.cpp
  starpu/unit_starpu_task_runner_runloop_and_inflight.cpp
  starpu/unit_tensor_builder_from_starpu.cpp
  starpu/robustness/edgecases_starpu_setup.cpp
  starpu/robustness/edgecases_starpu_task_runner.cpp
  starpu/robustness/edgecases_tensor_builder_from_starpu.cpp)

add_library(
  unit_monitoring_tests_obj OBJECT
  monitoring/cpu_usage_provider.cpp monitoring/gpu_metrics_sampling.cpp
  monitoring/queue_size_gauge.cpp monitoring/unit_congestion_monitor.cpp
  monitoring/unit_metrics.cpp)

add_library(
  unit_utils_tests_obj OBJECT
  utils/unit_client_utils.cpp
  utils/unit_datatype_utils.cpp
  utils/unit_input_generator.cpp
  utils/unit_logger.cpp
  utils/unit_batching_trace_logger.cpp
  utils/unit_nvtx.cpp
  utils/unit_perf_observer.cpp
  utils/unit_time_utils.cpp
  utils/unit_runtime_config.cpp
  utils/unit_config_loader.cpp
  utils/unit_exception_classification.cpp
  utils/unit_transparent_hash.cpp
  utils/robustness/edgecases_client_utils.cpp)
