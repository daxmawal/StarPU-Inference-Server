#pragma once

#include <chrono>
#include <iostream>
#include <utility>

#include "../test_helpers.hpp"

namespace starpu_server {
inline auto
make_job_with_callback(
    bool& called, std::vector<torch::Tensor>& results,
    double& latency) -> std::shared_ptr<InferenceJob>
{
  auto job = std::make_shared<InferenceJob>();
  job->set_on_complete([&](const std::vector<torch::Tensor>& r, double l) {
    called = true;
    results = r;
    latency = l;
  });
  return job;
}

struct CallbackProbe {
  bool called = false;
  std::vector<torch::Tensor> results;
  double latency = 0.0;
  std::shared_ptr<InferenceJob> job;
};

inline CallbackProbe
make_callback_probe()
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

struct TestBuffers {
  float input_data[3];
  float output_data[3];
  starpu_variable_interface input_iface;
  starpu_variable_interface output_iface;
  void* buffers[2];
};

inline TestBuffers
make_test_buffers()
{
  TestBuffers t{};
  t.input_data[0] = 1.0f;
  t.input_data[1] = 2.0f;
  t.input_data[2] = 3.0f;
  t.output_data[0] = t.output_data[1] = t.output_data[2] = 0.0f;
  t.input_iface = starpu_server::make_variable_interface(t.input_data);
  t.output_iface = starpu_server::make_variable_interface(t.output_data);
  t.buffers[0] = &t.input_iface;
  t.buffers[1] = &t.output_iface;
  return t;
}

struct TimingParams {
  starpu_server::InferenceParams params;
  starpu_server::DeviceType executed_on = starpu_server::DeviceType::Unknown;
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  torch::jit::script::Module model;
};

inline TimingParams
setup_timing_params(int elements)
{
  TimingParams t{starpu_server::make_basic_params(elements)};
  t.params.device.executed_on = &t.executed_on;
  t.params.timing.codelet_start_time = &t.start_time;
  t.params.timing.codelet_end_time = &t.end_time;
  t.model = torch::jit::script::Module("m");
  t.model.define(R"JIT(
        def forward(self, x):
            return x + 1
    )JIT");
  t.params.models.model_cpu = &t.model;
  return t;
}
