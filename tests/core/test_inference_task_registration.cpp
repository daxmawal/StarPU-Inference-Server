#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"

TEST(InferenceTaskRegistration, UndefinedTensorThrows)
{
  torch::Tensor undefined_tensor;  // not defined

  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_job_id(0);
  job->set_input_tensors({undefined_tensor});
  job->set_input_types({at::kFloat});
  job->set_outputs_tensors({torch::zeros({1})});

  torch::jit::script::Module model_cpu{"m"};
  std::vector<torch::jit::script::Module> models_gpu;
  starpu_server::RuntimeConfig opts;
  starpu_server::InferenceTask task(
      nullptr, job, &model_cpu, &models_gpu, &opts);

  EXPECT_THROW(
      starpu_server::InferenceTask::safe_register_tensor_vector(
          undefined_tensor, "input"),
      starpu_server::StarPURegistrationException);
}
