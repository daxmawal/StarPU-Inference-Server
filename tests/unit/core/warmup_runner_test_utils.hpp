#pragma once

#include <gtest/gtest.h>
#include <torch/script.h>

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "inference_runner_test_utils.hpp"

template <class F>
auto
measure_ms(F&& function) -> long
{
  const auto start = std::chrono::steady_clock::now();
  function();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

inline auto
make_device_workers() -> std::map<int, std::vector<int32_t>>
{
  return {{0, {1, 2}}};
}

struct WarmupRunnerTestFixture {
  starpu_server::RuntimeConfig opts;
  std::unique_ptr<starpu_server::StarPUSetup> starpu;
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> outputs_ref;
  void init(bool use_cuda = false)
  {
    opts = starpu_server::RuntimeConfig{};
    opts.input_shapes = {{1}};
    opts.input_types = {at::kFloat};
    opts.use_cuda = use_cuda;

    starpu = std::make_unique<starpu_server::StarPUSetup>(opts);
    model_cpu = starpu_server::make_identity_model();
    models_gpu.clear();
    outputs_ref = {torch::zeros({1})};
  }
  auto make_runner() -> starpu_server::WarmupRunner
  {
    return starpu_server::WarmupRunner(
        opts, *starpu, model_cpu, models_gpu, outputs_ref);
  }
};

class WarmupRunnerTest : public ::testing::Test,
                         public WarmupRunnerTestFixture {
 protected:
  std::unique_ptr<starpu_server::WarmupRunner> runner;

  void SetUp() override
  {
    init(false);
    runner = std::make_unique<starpu_server::WarmupRunner>(
        opts, *starpu, model_cpu, models_gpu, outputs_ref);
  }

  void init(bool use_cuda)
  {
    WarmupRunnerTestFixture::init(use_cuda);
    runner = std::make_unique<starpu_server::WarmupRunner>(
        opts, *starpu, model_cpu, models_gpu, outputs_ref);
  }
};
