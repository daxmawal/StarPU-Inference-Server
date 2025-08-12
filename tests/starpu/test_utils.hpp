#pragma once

#include <starpu.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "test_helpers.hpp"

namespace starpu_server {
inline auto
make_job_with_callback(
    bool& called, std::vector<torch::Tensor>& results,
    double& latency) -> std::shared_ptr<InferenceJob>
{
  auto job = std::make_shared<InferenceJob>();
  job->set_on_complete([&](const std::vector<torch::Tensor>& res, double lat) {
    called = true;
    results = res;
    latency = lat;
  });
  return job;
}

struct CallbackProbe {
  bool called = false;
  std::vector<torch::Tensor> results;
  double latency = 0.0;
  std::shared_ptr<InferenceJob> job;
};

inline auto
make_callback_probe() -> CallbackProbe
{
  CallbackProbe probe{};
  probe.job =
      make_job_with_callback(probe.called, probe.results, probe.latency);
  return probe;
}

template <typename F>
inline auto
capture_stdout(F&& func) -> std::string
{
  CaptureStream capture{std::cout};
  std::forward<F>(func)();
  return capture.str();
}
}  // namespace starpu_server

struct StarpuRuntimeGuard {
  StarpuRuntimeGuard()
  {
    if (starpu_init(nullptr) != 0) {
      throw std::runtime_error("StarPU initialization failed");
    }
  }
  ~StarpuRuntimeGuard()
  {
    starpu_shutdown();
#ifndef NDEBUG
    assert(starpu_is_initialized() == 0 && "StarPU shutdown failed");
#endif
  }
  StarpuRuntimeGuard(const StarpuRuntimeGuard&) = delete;
  auto operator=(const StarpuRuntimeGuard&) -> StarpuRuntimeGuard& = delete;
  StarpuRuntimeGuard(StarpuRuntimeGuard&&) = delete;
  auto operator=(StarpuRuntimeGuard&&) -> StarpuRuntimeGuard& = delete;
};

struct TestBuffers {
  std::array<float, 3> input_data;
  std::array<float, 3> output_data;
  starpu_variable_interface input_iface;
  starpu_variable_interface output_iface;
  std::array<void*, 2> buffers;
};

inline auto
make_test_buffers() -> TestBuffers
{
  TestBuffers buf{};
  buf.input_data[0] = 1.0F;
  buf.input_data[1] = 2.0F;
  buf.input_data[2] = 3.0F;
  buf.output_data[0] = buf.output_data[1] = buf.output_data[2] = 0.0F;
  buf.input_iface =
      starpu_server::make_variable_interface(buf.input_data.data());
  buf.output_iface =
      starpu_server::make_variable_interface(buf.output_data.data());
  buf.buffers[0] = &buf.input_iface;
  buf.buffers[1] = &buf.output_iface;
  return buf;
}

struct TimingParams {
  starpu_server::InferenceParams params;
  starpu_server::DeviceType executed_on = starpu_server::DeviceType::Unknown;
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  torch::jit::script::Module model;
};

inline auto
setup_timing_params(int elements) -> TimingParams
{
  TimingParams time{starpu_server::make_basic_params(elements)};
  time.params.device.executed_on = &time.executed_on;
  time.params.timing.codelet_start_time = &time.start_time;
  time.params.timing.codelet_end_time = &time.end_time;
  time.model = torch::jit::script::Module("m");
  time.model.define(R"JIT(
        def forward(self, x):
            return x + 1
    )JIT");
  time.params.models.model_cpu = &time.model;
  return time;
}
