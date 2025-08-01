#include <gtest/gtest.h>

#define private public
#include "core/warmup.hpp"
#undef private

using namespace starpu_server;

static auto
make_identity_model() -> torch::jit::script::Module
{
  torch::jit::script::Module m{"m"};
  m.define(R"JIT(
        def forward(self, x):
            return x
    )JIT");
  return m;
}

TEST(WarmupRunnerTest, ClientWorkerNegativeIterations)
{
  RuntimeConfig opts;
  opts.input_shapes = {{1}};
  opts.input_types = {at::kFloat};
  opts.use_cuda = false;

  StarPUSetup starpu(opts);
  auto model_cpu = make_identity_model();
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> outputs_ref = {torch::zeros({1})};

  WarmupRunner runner(opts, starpu, model_cpu, models_gpu, outputs_ref);

  std::map<int, std::vector<int32_t>> device_workers;
  InferenceQueue queue;

  EXPECT_THROW(
      runner.client_worker(device_workers, queue, -1), std::invalid_argument);
}