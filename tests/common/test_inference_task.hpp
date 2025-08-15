#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include <limits>
#include <vector>

#include "core/inference_task.hpp"
#include "test_helpers.hpp"
#include "utils/exceptions.hpp"

class InferenceTaskTest : public ::testing::Test {
 protected:
  static auto make_job(int job_id, size_t num_inputs, bool set_outputs = true)
      -> std::shared_ptr<starpu_server::InferenceJob>
  {
    auto job = std::make_shared<starpu_server::InferenceJob>();
    job->set_job_id(job_id);
    std::vector<torch::Tensor> inputs(num_inputs);
    std::vector<at::ScalarType> types(num_inputs, at::kFloat);
    for (size_t i = 0; i < num_inputs; ++i) {
      inputs[i] = torch::ones({1}, torch::TensorOptions().dtype(at::kFloat));
    }
    job->set_input_tensors(inputs);
    job->set_input_types(types);
    if (set_outputs) {
      job->set_outputs_tensors({torch::zeros({1})});
    }
    return job;
  }

  auto make_task(
      const std::shared_ptr<starpu_server::InferenceJob>& job,
      size_t num_gpu_models = 0) -> starpu_server::InferenceTask
  {
    model_cpu_ = starpu_server::make_add_one_model();
    models_gpu_.clear();
    for (size_t i = 0; i < num_gpu_models; ++i) {
      models_gpu_.push_back(starpu_server::make_add_one_model());
    }
    opts_ = starpu_server::RuntimeConfig{};
    return starpu_server::InferenceTask(
        nullptr, job, &model_cpu_, &models_gpu_, &opts_);
  }

  torch::jit::script::Module model_cpu_;
  std::vector<torch::jit::script::Module> models_gpu_;
  starpu_server::RuntimeConfig opts_;
};
