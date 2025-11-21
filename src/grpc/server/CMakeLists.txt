set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(CORE_DIR ${SRC_DIR}/core)
set(UTILS_DIR ${SRC_DIR}/utils)
set(STARPU_TASK_WORKER_DIR ${SRC_DIR}/starpu_task_worker)
set(MONITORING_DIR ${SRC_DIR}/monitoring)

add_executable(starpu_server)
target_sources(
  starpu_server
  PRIVATE ${CORE_DIR}/inference_task.cpp
          ${CORE_DIR}/input_slot_pool.cpp
          ${CORE_DIR}/output_slot_pool.cpp
          ${CORE_DIR}/starpu_setup.cpp
          ${CORE_DIR}/tensor_builder.cpp
          ${CORE_DIR}/warmup.cpp
          ${CORE_DIR}/inference_session.cpp
          ${CORE_DIR}/inference_runner.cpp
          ${UTILS_DIR}/client_utils.cpp
          ${UTILS_DIR}/batching_trace_logger.cpp
          ${UTILS_DIR}/inference_validator.cpp
          ${UTILS_DIR}/perf_observer.cpp
          ${UTILS_DIR}/time_utils.cpp
          ${UTILS_DIR}/config_loader.cpp
          ${STARPU_TASK_WORKER_DIR}/starpu_task_worker.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/inference_service.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/server_main.cpp
          ${MONITORING_DIR}/metrics.cpp)

set_target_properties(starpu_server PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                               ${CMAKE_BINARY_DIR})
target_compile_features(starpu_server PRIVATE cxx_std_23)
target_compile_definitions(starpu_server PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_include_directories(
  starpu_server PRIVATE ${SRC_DIR} ${CORE_DIR} ${UTILS_DIR}
                        ${STARPU_TASK_WORKER_DIR} ${MONITORING_DIR})

target_link_libraries(
  starpu_server
  PRIVATE Torch::Torch
          inference_proto
          Threads::Threads
          ${Protobuf_LIBRARIES}
          utf8_range::utf8_range
          utf8_range::utf8_validity
          ${ABSL_LIBS}
          absl::log_internal_message
          absl::log_internal_check_op
          absl::raw_logging_internal
          StarPU::starpu
          prometheus-cpp::pull
          yaml-cpp::yaml-cpp
          ${HWLOC_LIBRARY})

if(TARGET gRPC::grpc++_reflection)
  target_link_libraries(starpu_server PRIVATE gRPC::grpc++_reflection)
elseif(TARGET grpc++_reflection)
  target_link_libraries(starpu_server PRIVATE grpc++_reflection)
else()
  message(
    WARNING
      "gRPC reflection library not found; starpu_server will be built without reflection."
  )
endif()

if(STARPU_HAVE_NVML)
  target_compile_definitions(starpu_server PRIVATE STARPU_HAVE_NVML=1)
  target_include_directories(starpu_server PRIVATE ${NVML_INCLUDE_DIR})
  target_link_libraries(starpu_server PRIVATE ${NVML_LIBRARY})
endif()

if(HAVE_NVTX)
  target_compile_definitions(starpu_server PRIVATE HAVE_NVTX=1)
  if(CUDAToolkit_FOUND)
    target_include_directories(starpu_server
                               PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  endif()
endif()
