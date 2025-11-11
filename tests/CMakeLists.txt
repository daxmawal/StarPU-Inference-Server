set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(UTILS_DIR ${SRC_DIR}/utils)
set(CORE_DIR ${SRC_DIR}/core)
set(TASK_WORKER_DIR ${SRC_DIR}/starpu_task_worker)
set(GRPC_SERVER_DIR ${SRC_DIR}/grpc/server)
set(GRPC_CLIENT_DIR ${SRC_DIR}/grpc/client)
set(TESTS_DIR ${PROJECT_SOURCE_DIR}/tests)

add_subdirectory(unit)
add_subdirectory(e2e)
add_subdirectory(integration)

add_executable(
  all_tests
  ${CORE_DIR}/inference_task.cpp
  ${CORE_DIR}/input_slot_pool.cpp
  ${CORE_DIR}/output_slot_pool.cpp
  ${CORE_DIR}/warmup.cpp
  ${CORE_DIR}/tensor_builder.cpp
  ${CORE_DIR}/starpu_setup.cpp
  ${CORE_DIR}/inference_session.cpp
  ${CORE_DIR}/inference_runner.cpp
  ${SRC_DIR}/monitoring/metrics.cpp
  ${GRPC_CLIENT_DIR}/inference_client.cpp
  ${TASK_WORKER_DIR}/starpu_task_worker.cpp
  ${GRPC_CLIENT_DIR}/client_args.cpp
  ${UTILS_DIR}/inference_validator.cpp
  ${UTILS_DIR}/client_utils.cpp
  ${UTILS_DIR}/batching_trace_logger.cpp
  ${UTILS_DIR}/perf_observer.cpp
  ${UTILS_DIR}/time_utils.cpp
  ${UTILS_DIR}/config_loader.cpp)

set_target_properties(all_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                           ${CMAKE_BINARY_DIR})

target_compile_features(all_tests PRIVATE cxx_std_23)
target_compile_definitions(all_tests PRIVATE _GLIBCXX_USE_CXX11_ABI=1
                                             HAVE_NVTX=1)

target_include_directories(
  all_tests
  PRIVATE ${TESTS_DIR}
          ${TESTS_DIR}/common
          ${SRC_DIR}
          ${UTILS_DIR}
          ${CORE_DIR}
          ${TASK_WORKER_DIR}
          ${GRPC_SERVER_DIR}
          ${GRPC_CLIENT_DIR})

target_link_libraries(
  all_tests
  PRIVATE unit_tests
          e2e_tests
          integration_tests
          gtest
          gtest_main
          Torch::Torch
          inference_proto
          gRPC::grpc++
          ${Protobuf_LIBRARIES}
          Threads::Threads
          ${ABSL_LIBS}
          StarPU::starpu
          prometheus-cpp::pull
          yaml-cpp::yaml-cpp
          ${HWLOC_LIBRARY})

if(STARPU_HAVE_NVML)
  target_compile_definitions(all_tests PRIVATE STARPU_HAVE_NVML=1)
  target_include_directories(all_tests PRIVATE ${NVML_INCLUDE_DIR})
  target_link_libraries(all_tests PRIVATE ${NVML_LIBRARY})
endif()

if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(all_tests PRIVATE -fprofile-arcs -ftest-coverage -O0
                                           -g)
  target_link_options(all_tests PRIVATE -fprofile-arcs -ftest-coverage -O0 -g)
endif()

add_test(NAME all_tests COMMAND all_tests)
