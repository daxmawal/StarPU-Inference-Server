#include "inference_task.hpp"

#include <ATen/core/ScalarType.h>

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "args_parser.hpp"
#include "exceptions.hpp"
#include "inference_params.hpp"
#include "inference_runner.hpp"
#include "starpu_setup.hpp"

// =============================================================================
// Constructors
// =============================================================================
InferenceTask::InferenceTask(
    StarPUSetup& starpu, std::shared_ptr<InferenceJob> job,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const ProgramOptions& opts)
    : starpu_(starpu), job_(std::move(job)), model_cpu_(model_cpu),
      models_gpu_(models_gpu), opts_(opts)
{
}

InferenceCallbackContext::InferenceCallbackContext(
    std::shared_ptr<InferenceJob> job_,
    std::shared_ptr<InferenceParams> params_, const ProgramOptions& opts_,
    unsigned int id_, std::vector<starpu_data_handle_t> inputs_,
    std::vector<starpu_data_handle_t> outputs_)
    : job(std::move(job_)), inference_params(std::move(params_)), opts(opts_),
      id(id_), inputs_handles(std::move(inputs_)),
      outputs_handles(std::move(outputs_))
{
}

// =============================================================================
// Submission
// =============================================================================
void
InferenceTask::submit()
{
  if (!job_) {
    throw InvalidInferenceJobException("[ERROR] Job is null.");
  }

  auto inputs_handles = register_inputs_handles(job_->input_tensors);
  auto outputs_handles = register_outputs_handles(job_->outputs_tensors);
  const std::shared_ptr<InferenceParams> inference_params =
      create_inference_params();

  auto ctx = std::make_shared<InferenceCallbackContext>(
      job_, inference_params, opts_, job_->job_id, inputs_handles,
      outputs_handles);

  starpu_task* task = create_task(inputs_handles, outputs_handles, ctx);

  job_->timing_info.before_starpu_submitted_time =
      std::chrono::high_resolution_clock::now();

  const int ret = starpu_task_submit(task);
  if (ret != 0) {
    cleanup(ctx);
    throw StarPUTaskSubmissionException(
        "[ERROR] StarPU task submission failed (code " + std::to_string(ret) +
        ").");
  }
}

// =============================================================================
// Inference parameter construction
// =============================================================================
auto
InferenceTask::create_inference_params() -> std::shared_ptr<InferenceParams>
{
  const size_t num_inputs = job_->input_tensors.size();
  if (num_inputs > InferLimits::MaxInputs) {
    throw InferenceExecutionException(
        "Too many input tensors: max is " +
        std::to_string(InferLimits::MaxInputs));
  }

  auto params = std::make_shared<InferenceParams>();
  params->models.model_cpu = &model_cpu_;
  params->models.num_models_gpu = models_gpu_.size();

  if (models_gpu_.size() > InferLimits::MaxModelsGPU) {
    throw std::runtime_error(
        "Too many GPU models for the current configuration.");
  }
  for (size_t i = 0; i < models_gpu_.size(); ++i) {
    params->models.models_gpu[i] = &models_gpu_[i];
  }

  params->num_inputs = num_inputs;
  params->num_outputs = job_->outputs_tensors.size();
  params->job_id = job_->job_id;

  params->device.executed_on = &job_->executed_on;
  params->device.device_id = &job_->device_id;
  params->device.worker_id = &job_->worker_id;

  params->timing.codelet_start_time = &job_->timing_info.codelet_start_time;
  params->timing.codelet_end_time = &job_->timing_info.codelet_end_time;
  params->timing.inference_start_time = &job_->timing_info.inference_start_time;

  std::copy_n(
      job_->input_types.begin(), num_inputs,
      params->layout.input_types.begin());

  for (size_t i = 0; i < num_inputs; ++i) {
    const auto& tensor = job_->input_tensors[i];
    const int64_t dim = tensor.dim();
    params->layout.num_dims[i] = dim;
    std::copy_n(tensor.sizes().data(), dim, params->layout.dims[i].begin());
  }

  params->verbosity = opts_.verbosity;

  return params;
}

// =============================================================================
// StarPU data registration
// =============================================================================
auto
InferenceTask::safe_register_tensor_vector(
    const torch::Tensor& tensor,
    const std::string& label) -> starpu_data_handle_t
{
  if (!tensor.defined() || tensor.data_ptr() == nullptr) {
    throw StarPURegistrationException("Tensor '" + label + "' is invalid.");
  }

  starpu_data_handle_t handle = nullptr;

  starpu_vector_data_register(
      &handle, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(tensor.data_ptr()),
      static_cast<size_t>(static_cast<uint64_t>(tensor.numel())),
      static_cast<size_t>(static_cast<uint64_t>(tensor.element_size())));

  if (handle == nullptr) {
    throw StarPURegistrationException(
        "Failed to register tensor '" + label + "'.");
  }

  return handle;
}

auto
InferenceTask::register_inputs_handles(
    const std::vector<torch::Tensor>& input_tensors)
    -> std::vector<starpu_data_handle_t>
{
  std::vector<starpu_data_handle_t> handles;
  handles.reserve(input_tensors.size());

  for (size_t i = 0; i < input_tensors.size(); ++i) {
    handles.push_back(safe_register_tensor_vector(
        input_tensors[i], "input[" + std::to_string(i) + "]"));
  }

  return handles;
}

auto
InferenceTask::register_outputs_handles(
    const std::vector<torch::Tensor>& outputs_tensors)
    -> std::vector<starpu_data_handle_t>
{
  std::vector<starpu_data_handle_t> handles;
  handles.reserve(outputs_tensors.size());

  for (size_t i = 0; i < outputs_tensors.size(); ++i) {
    handles.push_back(safe_register_tensor_vector(
        outputs_tensors[i], "input[" + std::to_string(i) + "]"));
  }

  return handles;
}

// =============================================================================
// StarPU task creation
// =============================================================================
auto
InferenceTask::create_task(
    const std::vector<starpu_data_handle_t>& inputs_handles,
    const std::vector<starpu_data_handle_t>& outputs_handles,
    std::shared_ptr<InferenceCallbackContext> ctx) -> starpu_task*
{
  const size_t num_inputs = inputs_handles.size();
  const size_t num_buffers = num_inputs + outputs_handles.size();

  auto* task = starpu_task_create();
  if (task == nullptr) {
    throw StarPUTaskCreationException("Failed to create StarPU task.");
  }

  task->nbuffers = static_cast<int>(num_buffers);
  task->cl = starpu_.get_codelet();
  task->synchronous = opts_.synchronous ? 1 : 0;
  task->cl_arg = ctx->inference_params.get();
  task->cl_arg_size = sizeof(InferenceParams);
  task->priority = STARPU_MAX_PRIO - ctx->job->job_id;

  task->dyn_handles = static_cast<starpu_data_handle_t*>(
      malloc(num_buffers * sizeof(starpu_data_handle_t)));
  task->dyn_modes = static_cast<starpu_data_access_mode*>(
      malloc(num_buffers * sizeof(starpu_data_access_mode)));

  if (!task->dyn_handles || !task->dyn_modes) {
    starpu_task_destroy(task);
    cleanup(ctx);
    throw MemoryAllocationException("Failed to allocate task buffers.");
  }

  for (size_t idx = 0; idx < num_inputs; ++idx) {
    task->dyn_handles[idx] = inputs_handles[idx];
    task->dyn_modes[idx] = STARPU_R;
  }

  for (size_t idx = num_inputs; idx < num_buffers; ++idx) {
    task->dyn_handles[idx] = outputs_handles[idx - num_inputs];
    task->dyn_modes[idx] = STARPU_W;
  }

  task->callback_func = InferenceTask::starpu_output_callback;
  ctx->self_keep_alive = ctx;
  task->callback_arg = ctx.get();

  if (job_->fixed_worker_id.has_value()) {
    task->workerid = job_->fixed_worker_id.value();
    task->execute_on_a_specific_worker = 1;
  }

  return task;
}

// =============================================================================
// Callbacks & Cleanup
// =============================================================================
void
InferenceTask::cleanup(std::shared_ptr<InferenceCallbackContext> ctx)
{
  if (ctx == nullptr) {
    return;
  }

  for (auto& handle : ctx->inputs_handles) {
    if (handle != nullptr) {
      starpu_data_unregister_submit(handle);
      handle == nullptr;
    }
  }

  for (auto& handle : ctx->outputs_handles) {
    if (handle != nullptr) {
      starpu_data_unregister_submit(handle);
      handle == nullptr;
    }
  }
}

void
InferenceTask::on_output_ready_and_cleanup(void* arg)
{
  auto* ctx = static_cast<InferenceCallbackContext*>(arg);

  for (auto& handle : ctx->outputs_handles) {
    if (handle != nullptr) {
      starpu_data_release(handle);
    }
  }

  auto ctx_sptr =
      std::static_pointer_cast<InferenceCallbackContext>(ctx->self_keep_alive);
  if (ctx_sptr) {
    InferenceTask::cleanup(ctx_sptr);
  }

  const auto end_time = std::chrono::high_resolution_clock::now();

  if (ctx->job) {
    const double latency_ms = std::chrono::duration<double, std::milli>(
                                  end_time - ctx->job->start_time)
                                  .count();

    ctx->job->timing_info.callback_end_time = end_time;

    if (ctx->job->on_complete) {
      ctx->job->on_complete(ctx->job->outputs_tensors, latency_ms);
    }
  }

  ctx->self_keep_alive.reset();
}

void
InferenceTask::starpu_output_callback(void* arg)
{
  try {
    auto* ctx = static_cast<InferenceCallbackContext*>(arg);
    ctx->job->timing_info.callback_start_time =
        std::chrono::high_resolution_clock::now();

    ctx->remaining_outputs_to_acquire =
        static_cast<int>(ctx->outputs_handles.size());

    for (auto& handle : ctx->outputs_handles) {
      if (handle != nullptr) {
        starpu_data_acquire_cb(
            handle, STARPU_R,
            [](void* cb_arg) {
              auto* cb_ctx = static_cast<InferenceCallbackContext*>(cb_arg);

              if (--cb_ctx->remaining_outputs_to_acquire == 0) {
                try {
                  InferenceTask::on_output_ready_and_cleanup(cb_ctx);
                }
                catch (...) {
                  log_exception("on_output_ready");
                }
              }
            },
            ctx);
      } else {
        if (--ctx->remaining_outputs_to_acquire == 0) {
          InferenceTask::on_output_ready_and_cleanup(ctx);
        }
      }
    }
  }
  catch (...) {
    log_exception("starpu_output_callback");
  }
}

// =============================================================================
// Exception handling
// =============================================================================
void
InferenceTask::log_exception(const std::string& context)
{
  try {
    throw;
  }
  catch (const InferenceExecutionException& e) {
    log_error("InferenceExecutionException in " + context + ": " + e.what());
  }
  catch (const StarPUTaskSubmissionException& e) {
    log_error("StarPU submission error in " + context + ": " + e.what());
  }
  catch (const std::exception& e) {
    log_error("std::exception in " + context + ": " + e.what());
  }
  catch (...) {
    log_error("Unknown exception in " + context + ".");
  }
}
