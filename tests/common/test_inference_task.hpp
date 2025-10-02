#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include <limits>
#include <memory>
#include <utility>
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
      job->set_output_tensors({torch::zeros({1})});
    }
    return job;
  }

  auto make_task(
      const std::shared_ptr<starpu_server::InferenceJob>& job,
      size_t num_gpu_models = 0,
      const starpu_server::InferenceTaskDependencies* dependencies = nullptr)
      -> starpu_server::InferenceTask
  {
    model_cpu_ = starpu_server::make_add_one_model();
    models_gpu_.clear();
    for (size_t i = 0; i < num_gpu_models; ++i) {
      models_gpu_.push_back(starpu_server::make_add_one_model());
    }
    opts_ = starpu_server::RuntimeConfig{};
    const auto& deps = dependencies != nullptr
                           ? *dependencies
                           : starpu_server::kDefaultInferenceTaskDependencies;
    return starpu_server::InferenceTask(
        nullptr, job, &model_cpu_, &models_gpu_, &opts_, deps);
  }

  torch::jit::script::Module model_cpu_;
  std::vector<torch::jit::script::Module> models_gpu_;
  starpu_server::RuntimeConfig opts_;
};

inline auto
make_callback_context(
    std::shared_ptr<starpu_server::InferenceJob> job = nullptr,
    const starpu_server::RuntimeConfig* opts = nullptr,
    std::vector<starpu_data_handle_t> inputs = {},
    std::vector<starpu_data_handle_t> outputs = {},
    const starpu_server::InferenceTaskDependencies* dependencies = nullptr,
    std::shared_ptr<starpu_server::InferenceParams> params = nullptr,
    int id = 0) -> std::shared_ptr<starpu_server::InferenceCallbackContext>
{
  auto ctx = std::make_shared<starpu_server::InferenceCallbackContext>(
      std::move(job), std::move(params), opts, id, std::move(inputs),
      std::move(outputs));
  ctx->dependencies = dependencies;
  return ctx;
}
