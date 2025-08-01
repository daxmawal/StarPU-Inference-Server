set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(UTILS_DIR ${SRC_DIR}/utils)
set(CORE_DIR ${SRC_DIR}/core)
set(CLI_DIR ${SRC_DIR}/cli)
set(TASK_WORKER_DIR ${SRC_DIR}/starpu_task_worker)
set(GRPC_SERVER_DIR ${SRC_DIR}/grpc/server)

add_executable(
  unit_tests
  test_datatype_utils.cpp
  test_time_utils.cpp
  test_device_type.cpp
  test_transparent_hash.cpp
  test_warmup_runner.cpp
  test_input_generator.cpp
  test_logger.cpp
  test_input_generator_utils.cpp
  test_inference_queue.cpp
  test_inference_validator.cpp
  test_inference_service.cpp
  test_tensor_builder.cpp
  test_tensor_builder_from_starpu.cpp
  test_starpu_task_runner.cpp
  test_client_utils.cpp
  test_args_parser.cpp
  test_inference_task_registration.cpp
  test_inference_runner_utils.cpp
  test_inference_service_impl_methods.cpp
  test_log_exception.cpp
  test_inference_task.cpp
  ${CLI_DIR}/args_parser.cpp
  ${CORE_DIR}/inference_task.cpp
  ${CORE_DIR}/warmup.cpp
  ${CORE_DIR}/tensor_builder.cpp
  ${CORE_DIR}/starpu_setup.cpp
  ${GRPC_SERVER_DIR}/inference_service.cpp
  ${TASK_WORKER_DIR}/starpu_task_worker.cpp
  ${UTILS_DIR}/inference_validator.cpp
  ${UTILS_DIR}/client_utils.cpp
  ${UTILS_DIR}/time_utils.cpp)

set_target_properties(unit_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                            ${CMAKE_BINARY_DIR})

target_compile_features(unit_tests PRIVATE cxx_std_23)
target_compile_definitions(unit_tests PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_include_directories(
  unit_tests PRIVATE ${SRC_DIR} ${UTILS_DIR} ${CLI_DIR} ${CORE_DIR}
                     ${TASK_WORKER_DIR} ${GRPC_SERVER_DIR})

target_link_libraries(
  unit_tests
  PRIVATE gtest
          gtest_main
          Torch::Torch
          inference_proto
          gRPC::grpc++
          ${Protobuf_LIBRARIES}
          Threads::Threads
          ${ABSL_LIBS}
          StarPU::starpu)

if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(unit_tests PRIVATE -fprofile-arcs -ftest-coverage -O0
                                            -g)
  target_link_options(unit_tests PRIVATE -fprofile-arcs -ftest-coverage -O0 -g)
endif()

add_test(NAME unit_tests COMMAND unit_tests)
