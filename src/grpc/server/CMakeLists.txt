set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(CORE_DIR ${SRC_DIR}/core)
set(UTILS_DIR ${SRC_DIR}/utils)
set(STARPU_TASK_WORKER_DIR ${SRC_DIR}/starpu_task_worker)
set(MONITORING_DIR ${SRC_DIR}/monitoring)

add_executable(grpc_server)
target_sources(
  grpc_server
  PRIVATE ${CORE_DIR}/inference_task.cpp
          ${CORE_DIR}/starpu_setup.cpp
          ${CORE_DIR}/tensor_builder.cpp
          ${CORE_DIR}/warmup.cpp
          ${UTILS_DIR}/client_utils.cpp
          ${UTILS_DIR}/inference_validator.cpp
          ${UTILS_DIR}/time_utils.cpp
          ${UTILS_DIR}/config_loader.cpp
          ${STARPU_TASK_WORKER_DIR}/starpu_task_worker.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/inference_service.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/server_main.cpp
          ${MONITORING_DIR}/metrics.cpp)

set_target_properties(grpc_server PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                             ${CMAKE_BINARY_DIR})
target_compile_features(grpc_server PRIVATE cxx_std_23)
target_compile_definitions(grpc_server PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_include_directories(
  grpc_server PRIVATE ${SRC_DIR} ${CORE_DIR} ${UTILS_DIR}
                      ${STARPU_TASK_WORKER_DIR} ${MONITORING_DIR})

target_link_libraries(
  grpc_server
  PRIVATE Torch::Torch
          inference_proto
          gRPC::grpc++_reflection
          Threads::Threads
          ${Protobuf_LIBRARIES}
          utf8_range::utf8_range
          utf8_range::utf8_validity
          ${ABSL_LIBS}
          StarPU::starpu
          prometheus-cpp::pull
          yaml-cpp::yaml-cpp)
