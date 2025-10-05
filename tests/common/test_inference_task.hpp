#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "core/inference_task.hpp"
#include "core/output_slot_pool.hpp"
#include "test_helpers.hpp"
#include "test_utils.hpp"
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

inline auto
make_runtime_config_with_single_output() -> starpu_server::RuntimeConfig
{
  starpu_server::RuntimeConfig opts;
  starpu_server::ModelConfig model;
  model.name = "test_model";
  starpu_server::TensorConfig output;
  output.name = "output";
  output.dims = {1};
  output.type = at::kFloat;
  model.outputs.push_back(output);
  opts.models.push_back(std::move(model));
  return opts;
}

struct OutputContextFixtureOptions {
  std::optional<float> sentinel_value{};
  std::function<void(std::vector<torch::Tensor>&)> mutate_job_outputs{};
  std::function<void(starpu_server::OutputSlotPool&, int)> on_finished{};
};

struct OutputContextFixture {
  explicit OutputContextFixture(OutputContextFixtureOptions options = {})
      : starpu_guard(), opts(make_runtime_config_with_single_output()),
        pool(opts, 1), slot_id(pool.acquire()),
        job(std::make_shared<starpu_server::InferenceJob>()),
        ctx(make_callback_context(job, &opts))
  {
    std::vector<torch::Tensor> outputs{
        torch::zeros({1}, torch::TensorOptions().dtype(at::kFloat))};
    if (options.mutate_job_outputs) {
      options.mutate_job_outputs(outputs);
    }
    job->set_output_tensors(outputs);

    ctx->output_pool = &pool;
    ctx->output_slot_id = slot_id;
    ctx->self_keep_alive = ctx;

    auto on_finished = options.on_finished;
    ctx->on_finished = [on_finished, pool_ptr = &pool, slot = slot_id]() {
      if (on_finished) {
        on_finished(*pool_ptr, slot);
      } else {
        pool_ptr->release(slot);
      }
    };

    if (options.sentinel_value.has_value()) {
      const auto& base_ptrs = pool.base_ptrs(slot_id);
      if (!base_ptrs.empty() && base_ptrs[0] != nullptr) {
        *static_cast<float*>(base_ptrs[0]) = *options.sentinel_value;
      }
    }
  }

  StarpuRuntimeGuard starpu_guard;
  starpu_server::RuntimeConfig opts;
  starpu_server::OutputSlotPool pool;
  int slot_id;
  std::shared_ptr<starpu_server::InferenceJob> job;
  std::shared_ptr<starpu_server::InferenceCallbackContext> ctx;
};
