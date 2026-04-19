set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(UTILS_DIR ${SRC_DIR}/utils)
set(CORE_DIR ${SRC_DIR}/core)
set(TASK_WORKER_DIR ${SRC_DIR}/starpu_task_worker)
set(GRPC_SERVER_DIR ${SRC_DIR}/grpc/server)
set(GRPC_CLIENT_DIR ${SRC_DIR}/grpc/client)
set(MONITORING_DIR ${SRC_DIR}/monitoring)
set(TESTS_DIR ${PROJECT_SOURCE_DIR}/tests)

function(starpu_configure_test_compile_target target_name)
  target_compile_features(${target_name} PRIVATE cxx_std_23)
  target_compile_definitions(${target_name} PRIVATE _GLIBCXX_USE_CXX11_ABI=1
                                                    STARPU_TESTING=1)
  target_include_directories(
    ${target_name}
    PRIVATE ${TESTS_DIR}
            ${TESTS_DIR}/common
            ${SRC_DIR}
            ${UTILS_DIR}
            ${CORE_DIR}
            ${TASK_WORKER_DIR}
            ${GRPC_SERVER_DIR}
            ${GRPC_CLIENT_DIR})
  target_link_libraries(
    ${target_name}
    PRIVATE gtest
            Torch::Torch
            inference_proto
            Threads::Threads
            StarPU::starpu
            prometheus-cpp::pull
            yaml-cpp::yaml-cpp
            HWLOC::hwloc)

  if(TARGET gRPC::grpc++_reflection)
    target_link_libraries(${target_name} PRIVATE gRPC::grpc++_reflection)
  elseif(TARGET grpc++_reflection)
    target_link_libraries(${target_name} PRIVATE grpc++_reflection)
  endif()

  if(STARPU_HAVE_NVML)
    target_compile_definitions(${target_name} PRIVATE STARPU_HAVE_NVML=1)
    target_include_directories(${target_name} PRIVATE ${NVML_INCLUDE_DIR})
    target_link_libraries(${target_name} PRIVATE ${NVML_LIBRARY})
  endif()

  if(HAVE_NVTX)
    target_compile_definitions(${target_name} PRIVATE HAVE_NVTX=1)
  endif()

  if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target_name} PRIVATE -fprofile-arcs
                                                  -ftest-coverage -O0 -g)
  endif()

  if(ENABLE_SANITIZERS)
    target_compile_options(
      ${target_name} PRIVATE -fsanitize=address,undefined
                             -fno-omit-frame-pointer -fno-sanitize-recover=all)
  endif()

  if(ENABLE_TSAN)
    target_compile_options(${target_name} PRIVATE -fsanitize=thread
                                                  -fno-omit-frame-pointer)
  endif()
endfunction()

function(starpu_configure_test_binary target_name)
  target_link_libraries(${target_name} PRIVATE starpu_test_runtime gtest_main)

  if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(${target_name} PRIVATE -fprofile-arcs -ftest-coverage
                        -O0 -g)
  endif()

  starpu_apply_sanitizers(${target_name})
  starpu_apply_tsan(${target_name})

  set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  ${CMAKE_BINARY_DIR})
endfunction()

function(starpu_add_domain_ctest target_name timeout_seconds)
  set(labels "${ARGN}")
  add_test(NAME ${target_name} COMMAND $<TARGET_FILE:${target_name}>)
  set_tests_properties(${target_name} PROPERTIES TIMEOUT ${timeout_seconds}
                                                 LABELS "${labels}")
endfunction()

add_subdirectory(unit)
add_subdirectory(e2e)
add_subdirectory(integration)

set(STARPU_TEST_RUNTIME_SOURCES
    ${CORE_DIR}/inference_runner.cpp
    ${CORE_DIR}/inference_task.cpp
    ${CORE_DIR}/input_slot_pool.cpp
    ${CORE_DIR}/output_slot_pool.cpp
    ${CORE_DIR}/tensor_builder.cpp
    ${CORE_DIR}/warmup.cpp
    ${CORE_DIR}/starpu_setup.cpp
    ${MONITORING_DIR}/metrics.cpp
    ${MONITORING_DIR}/congestion_monitor.cpp
    ${TASK_WORKER_DIR}/starpu_task_worker.cpp
    ${TASK_WORKER_DIR}/batch_capacity_policy.cpp
    ${TASK_WORKER_DIR}/batch_composition_policy.cpp
    ${TASK_WORKER_DIR}/batching_strategy.cpp
    ${TASK_WORKER_DIR}/batching_strategy_input_provider.cpp
    ${TASK_WORKER_DIR}/batch_collector_component.cpp
    ${TASK_WORKER_DIR}/result_dispatcher_component.cpp
    ${TASK_WORKER_DIR}/slot_manager_component.cpp
    ${GRPC_SERVER_DIR}/inference_service.cpp
    ${GRPC_SERVER_DIR}/inference_service_io.cpp
    ${GRPC_SERVER_DIR}/inference_service_metadata.cpp
    ${GRPC_SERVER_DIR}/inference_service_async.cpp
    ${GRPC_SERVER_DIR}/inference_service_async_server.cpp
    ${GRPC_CLIENT_DIR}/client_args.cpp
    ${GRPC_CLIENT_DIR}/inference_client.cpp
    ${UTILS_DIR}/client_utils.cpp
    ${UTILS_DIR}/batching_trace_logger.cpp
    ${UTILS_DIR}/perf_observer.cpp
    ${UTILS_DIR}/time_utils.cpp
    ${UTILS_DIR}/config_loader.cpp
    ${TESTS_DIR}/support/grpc/server/inference_service_test_internal.cpp
    ${TESTS_DIR}/support/grpc/server/inference_service_test_accessor.cpp
    ${TESTS_DIR}/support/output_slot_pool_test_hooks.cpp
    ${TESTS_DIR}/support/starpu_data_interface_override.cpp
    ${TESTS_DIR}/support/starpu_task_submit_override.cpp
    ${TESTS_DIR}/support/torch_cuda_device_count_override.cpp)

add_library(starpu_test_runtime STATIC ${STARPU_TEST_RUNTIME_SOURCES})
starpu_configure_test_compile_target(starpu_test_runtime)

set(STARPU_ALL_TEST_OBJECT_TARGETS
    unit_core_tests_obj
    unit_grpc_tests_obj
    unit_monitoring_tests_obj
    unit_starpu_tests_obj
    unit_utils_tests_obj
    integration_core_tests_obj
    integration_grpc_tests_obj
    integration_starpu_tests_obj
    integration_utils_tests_obj
    e2e_tests_obj)

foreach(test_object_target IN LISTS STARPU_ALL_TEST_OBJECT_TARGETS)
  starpu_configure_test_compile_target(${test_object_target})
endforeach()

function(starpu_add_domain_test_binary target_name)
  add_executable(${target_name})
  foreach(test_object_target IN LISTS ARGN)
    target_sources(${target_name}
                   PRIVATE $<TARGET_OBJECTS:${test_object_target}>)
  endforeach()
  starpu_configure_test_binary(${target_name})
endfunction()

starpu_add_domain_test_binary(all_tests ${STARPU_ALL_TEST_OBJECT_TARGETS})

starpu_add_domain_test_binary(unit_core_tests unit_core_tests_obj)
starpu_add_domain_test_binary(unit_grpc_tests unit_grpc_tests_obj)
starpu_add_domain_test_binary(unit_monitoring_tests unit_monitoring_tests_obj)
starpu_add_domain_test_binary(unit_starpu_tests unit_starpu_tests_obj)
starpu_add_domain_test_binary(unit_utils_tests unit_utils_tests_obj)
starpu_add_domain_test_binary(integration_core_tests integration_core_tests_obj)
starpu_add_domain_test_binary(integration_grpc_tests integration_grpc_tests_obj)
starpu_add_domain_test_binary(integration_starpu_tests
                              integration_starpu_tests_obj)
starpu_add_domain_test_binary(integration_utils_tests
                              integration_utils_tests_obj)
starpu_add_domain_test_binary(e2e_tests e2e_tests_obj)

starpu_add_domain_ctest(unit_core_tests 480 "unit;core")
starpu_add_domain_ctest(unit_grpc_tests 480 "unit;grpc")
starpu_add_domain_ctest(unit_monitoring_tests 480 "unit;monitoring")
starpu_add_domain_ctest(unit_starpu_tests 480 "unit;starpu")
starpu_add_domain_ctest(unit_utils_tests 480 "unit;utils")
starpu_add_domain_ctest(integration_core_tests 600 "integration;core")
starpu_add_domain_ctest(integration_grpc_tests 600 "integration;grpc")
starpu_add_domain_ctest(integration_starpu_tests 600 "integration;starpu")
starpu_add_domain_ctest(integration_utils_tests 600 "integration;utils")
starpu_add_domain_ctest(e2e_tests 600 "e2e")
