#pragma once

#include <gtest/gtest.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

#include "core/inference_params.hpp"
#include "core/inference_runner.hpp"
#include "starpu_task_worker/starpu_task_worker.hpp"
#include "test_utils.hpp"

class StarPUTaskRunnerFixture : public ::testing::Test {
 protected:
  starpu_server::InferenceQueue queue_;
  torch::jit::script::Module model_cpu_;
  std::vector<torch::jit::script::Module> models_gpu_;
  starpu_server::RuntimeConfig opts_;
  std::vector<starpu_server::InferenceResult> results_;
  std::mutex results_mutex_;
  std::atomic<int> completed_jobs_;
  std::condition_variable cv_;
  std::unique_ptr<starpu_server::StarPUSetup> starpu_setup_;
  starpu_server::StarPUTaskRunnerConfig config_{};
  std::unique_ptr<starpu_server::StarPUTaskRunner> runner_;
  void SetUp() override
  {
    completed_jobs_ = 0;
    starpu_setup_ = std::make_unique<starpu_server::StarPUSetup>(opts_);
    config_.queue = &queue_;
    config_.model_cpu = &model_cpu_;
    config_.models_gpu = &models_gpu_;
    config_.starpu = starpu_setup_.get();
    config_.opts = &opts_;
    config_.results = &results_;
    config_.results_mutex = &results_mutex_;
    config_.completed_jobs = &completed_jobs_;
    config_.all_done_cv = &cv_;
    runner_ = std::make_unique<starpu_server::StarPUTaskRunner>(config_);
  }
};
