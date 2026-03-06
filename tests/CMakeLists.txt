set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(UTILS_DIR ${SRC_DIR}/utils)
set(CORE_DIR ${SRC_DIR}/core)
set(TASK_WORKER_DIR ${SRC_DIR}/starpu_task_worker)
set(GRPC_SERVER_DIR ${SRC_DIR}/grpc/server)
set(GRPC_CLIENT_DIR ${SRC_DIR}/grpc/client)
set(MONITORING_DIR ${SRC_DIR}/monitoring)
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
  ${CORE_DIR}/starpu_setup.cpp
  ${MONITORING_DIR}/metrics.cpp
  ${MONITORING_DIR}/congestion_monitor.cpp
  ${TASK_WORKER_DIR}/starpu_task_worker.cpp
  ${GRPC_CLIENT_DIR}/client_args.cpp
  ${UTILS_DIR}/client_utils.cpp
  ${UTILS_DIR}/perf_observer.cpp
  ${UTILS_DIR}/time_utils.cpp
  ${UTILS_DIR}/config_loader.cpp)

set_target_properties(all_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                           ${CMAKE_BINARY_DIR})

target_compile_features(all_tests PRIVATE cxx_std_23)
target_compile_definitions(all_tests PRIVATE _GLIBCXX_USE_CXX11_ABI=1
                                             STARPU_TESTING=1)

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
          gtest_main
          Torch::Torch
          inference_proto
          Threads::Threads
          StarPU::starpu
          prometheus-cpp::pull
          yaml-cpp::yaml-cpp
          HWLOC::hwloc)

if(TARGET gRPC::grpc++_reflection)
  target_link_libraries(all_tests PRIVATE gRPC::grpc++_reflection)
elseif(TARGET grpc++_reflection)
  target_link_libraries(all_tests PRIVATE grpc++_reflection)
else()
  message(
    WARNING
      "gRPC reflection library not found; all_tests will be built without reflection."
  )
endif()

if(STARPU_HAVE_NVML)
  target_compile_definitions(all_tests PRIVATE STARPU_HAVE_NVML=1)
  target_include_directories(all_tests PRIVATE ${NVML_INCLUDE_DIR})
  target_link_libraries(all_tests PRIVATE ${NVML_LIBRARY})
endif()

if(HAVE_NVTX)
  target_compile_definitions(all_tests PRIVATE HAVE_NVTX=1)
endif()

if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(all_tests PRIVATE -fprofile-arcs -ftest-coverage -O0
                                           -g)
  target_link_options(all_tests PRIVATE -fprofile-arcs -ftest-coverage -O0 -g)
endif()

starpu_apply_sanitizers(all_tests)
starpu_apply_tsan(all_tests)

set(STARPU_TEST_FILTER_UNIT
    "*-*Integration*:*_Integration*:*E2E*:*EndToEndInference*")
set(STARPU_TEST_FILTER_INTEGRATION
    "*Integration*:*_Integration*:*EndToEndInference*")
set(STARPU_TEST_FILTER_E2E "*E2E*")

add_test(NAME unit_tests COMMAND all_tests
                                 --gtest_filter=${STARPU_TEST_FILTER_UNIT})
add_test(NAME integration_tests
         COMMAND all_tests --gtest_filter=${STARPU_TEST_FILTER_INTEGRATION})
add_test(NAME e2e_tests COMMAND all_tests
                                --gtest_filter=${STARPU_TEST_FILTER_E2E})
