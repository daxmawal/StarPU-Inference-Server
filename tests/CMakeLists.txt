set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(UTILS_DIR ${SRC_DIR}/utils)
set(CORE_DIR ${SRC_DIR}/core)
set(TASK_WORKER_DIR ${SRC_DIR}/starpu_task_worker)

add_executable(
  unit_tests
  test_datatype_utils.cpp
  test_time_utils.cpp
  test_device_type.cpp
  test_transparent_hash.cpp
  test_input_generator.cpp
  test_logger.cpp
  test_input_generator_utils.cpp
  test_inference_queue.cpp
  test_inference_validator.cpp
  test_tensor_builder.cpp
  test_client_utils.cpp
  ${CORE_DIR}/tensor_builder.cpp
  ${UTILS_DIR}/inference_validator.cpp
  ${UTILS_DIR}/client_utils.cpp
  ${UTILS_DIR}/time_utils.cpp)

set_target_properties(unit_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                            ${CMAKE_BINARY_DIR})

target_compile_features(unit_tests PRIVATE cxx_std_23)
target_compile_definitions(unit_tests PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_include_directories(unit_tests PRIVATE ${SRC_DIR} ${UTILS_DIR}
                                              ${CORE_DIR} ${TASK_WORKER_DIR})

target_link_libraries(unit_tests PRIVATE gtest gtest_main Torch::Torch
                                         StarPU::starpu)

add_test(NAME unit_tests COMMAND unit_tests)
