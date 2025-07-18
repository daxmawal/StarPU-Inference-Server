#include "inference_task.hpp"

#include <ATen/core/ScalarType.h>

#include <bit>
#include <chrono>
#include <cstddef>
#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "exceptions.hpp"
#include "inference_params.hpp"
#include "inference_runner.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"

namespace starpu_server {
// =============================================================================
// Constructor
// =============================================================================

InferenceTask::InferenceTask(
    StarPUSetup* starpu, std::shared_ptr<InferenceJob> job,
    torch::jit::script::Module* model_cpu,
    std::vector<torch::jit::script::Module>* models_gpu,
    const RuntimeConfig* opts)
    : starpu_(starpu), job_(std::move(job)), model_cpu_(model_cpu),
      models_gpu_(models_gpu), opts_(opts)
{
}

// =============================================================================
// InferenceCallbackContext
// Holds job metadata and StarPU handles needed during task lifecycle.
// Used as StarPU's callback context.
// =============================================================================

InferenceCallbackContext::InferenceCallbackContext(
    std::shared_ptr<InferenceJob> job_,
    std::shared_ptr<InferenceParams> params_, const RuntimeConfig* opts_,
    int id_, std::vector<starpu_data_handle_t> inputs_,
    std::vector<starpu_data_handle_t> outputs_)
    : job(std::move(job_)), inference_params(std::move(params_)), opts(opts_),
      id(id_), inputs_handles(std::move(inputs_)),
      outputs_handles(std::move(outputs_))
{
}

// =============================================================================
// Handle Registration
// Register input/output tensors as StarPU data handles.
// =============================================================================

auto
InferenceTask::prepare_input_handles() const
    -> std::vector<starpu_data_handle_t>
{
  return register_inputs_handles(job_->get_input_tensors());
}

auto
InferenceTask::prepare_output_handles() const
    -> std::vector<starpu_data_handle_t>
{
  return register_outputs_handles(job_->get_output_tensors());
}

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
      &handle, STARPU_MAIN_RAM, std::bit_cast<uintptr_t>(tensor.data_ptr()),
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
        input_tensors[i], std::format("input[{}]", i)));
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
        outputs_tensors[i], std::format("output[{}]", i)));
  }

  return handles;
}

// =============================================================================
// Inference Parameters Setup
// Fill inference configuration structures passed to StarPU.
// =============================================================================

auto
InferenceTask::create_context(
    const std::vector<starpu_data_handle_t>& inputs,
    const std::vector<starpu_data_handle_t>& outputs)
    -> std::shared_ptr<InferenceCallbackContext>
{
  auto params = create_inference_params();
  return std::make_shared<InferenceCallbackContext>(
      job_, std::move(params), opts_, job_->get_job_id(), inputs, outputs);
}

void
InferenceTask::check_limits(size_t num_inputs) const
{
  if (num_inputs > InferLimits::MaxInputs) {
    throw InferenceExecutionException(std::format(
        "Too many input tensors: max is {}", InferLimits::MaxInputs));
  }

  if (models_gpu_->size() > InferLimits::MaxModelsGPU) {
    throw TooManyGpuModelsException(
        "Too many GPU models for the current configuration.");
  }
}

auto
InferenceTask::create_inference_params() -> std::shared_ptr<InferenceParams>
{
  const size_t num_inputs = job_->get_input_tensors().size();
  check_limits(num_inputs);

  auto params = std::make_shared<InferenceParams>();

  fill_model_pointers(params);
  bind_runtime_job_info(params);
  fill_input_layout(params, num_inputs);

  params->num_inputs = num_inputs;
  params->num_outputs = job_->get_output_tensors().size();
  params->verbosity = opts_->verbosity;

  return params;
}

void
InferenceTask::fill_model_pointers(
    const std::shared_ptr<InferenceParams>& params) const
{
  params->models.model_cpu = model_cpu_;
  params->models.num_models_gpu = models_gpu_->size();

  for (size_t i = 0; i < models_gpu_->size(); ++i) {
    params->models.models_gpu.at(i) = &(models_gpu_->at(i));
  }
}

void
InferenceTask::bind_runtime_job_info(
    const std::shared_ptr<InferenceParams>& params) const
{
  params->job_id = job_->get_job_id();
  params->device.executed_on = &job_->get_executed_on();
  params->device.device_id = &job_->get_device_id();
  params->device.worker_id = &job_->get_worker_id();
  params->timing.codelet_start_time = &job_->timing_info().codelet_start_time;
  params->timing.codelet_end_time = &job_->timing_info().codelet_end_time;
  params->timing.inference_start_time =
      &job_->timing_info().inference_start_time;
}

void
InferenceTask::fill_input_layout(
    std::shared_ptr<InferenceParams>& params, size_t num_inputs) const
{
  std::copy_n(
      job_->get_input_types().begin(), num_inputs,
      params->layout.input_types.begin());

  for (size_t i = 0; i < num_inputs; ++i) {
    const auto& tensor = job_->get_input_tensors()[i];
    const int64_t dim = tensor.dim();
    if (dim > static_cast<int64_t>(InferLimits::MaxDims)) {
      throw InferenceExecutionException(std::format(
          "Input tensor has too many dimensions: max is {}",
          InferLimits::MaxDims));
    }
    params->layout.num_dims.at(i) = dim;
    std::copy_n(tensor.sizes().data(), dim, params->layout.dims.at(i).begin());
  }
}

// =============================================================================
// StarPU Task Construction
// Build, configure, and submit the StarPU task for execution.
// =============================================================================

void
InferenceTask::cleanup(const std::shared_ptr<InferenceCallbackContext>& ctx)
{
  if (ctx == nullptr) {
    return;
  }

  for (auto& handle : ctx->inputs_handles) {
    if (handle != nullptr) {
      starpu_data_unregister_submit(handle);
      handle = nullptr;
    }
  }

  for (auto& handle : ctx->outputs_handles) {
    if (handle != nullptr) {
      starpu_data_unregister_submit(handle);
      handle = nullptr;
    }
  }
}

void
InferenceTask::submit()
{
  if (!job_) {
    throw InvalidInferenceJobException("[ERROR] Job is null.");
  }

  auto inputs_handles = prepare_input_handles();
  auto outputs_handles = prepare_output_handles();
  auto ctx = create_context(inputs_handles, outputs_handles);
  starpu_task* task = create_task(inputs_handles, outputs_handles, ctx);

  job_->timing_info().before_starpu_submitted_time =
      std::chrono::high_resolution_clock::now();

  const int ret = starpu_task_submit(task);
  if (ret != 0) {
    cleanup(ctx);
    throw StarPUTaskSubmissionException(
        std::format("[ERROR] StarPU task submission failed (code {})", ret));
  }
}

auto
InferenceTask::create_task(
    const std::vector<starpu_data_handle_t>& inputs_handles,
    const std::vector<starpu_data_handle_t>& outputs_handles,
    const std::shared_ptr<InferenceCallbackContext>& ctx) -> starpu_task*
{
  const size_t num_inputs = inputs_handles.size();
  const size_t num_buffers = num_inputs + outputs_handles.size();

  auto* task = starpu_task_create();
  if (task == nullptr) {
    throw StarPUTaskCreationException("Failed to create StarPU task.");
  }

  task->nbuffers = static_cast<int>(num_buffers);
  task->cl = starpu_->get_codelet();
  task->synchronous = opts_->synchronous ? 1 : 0;
  task->cl_arg = ctx->inference_params.get();
  task->cl_arg_size = sizeof(InferenceParams);
  task->priority = STARPU_MAX_PRIO - ctx->job->get_job_id();

  InferenceTask::allocate_task_buffers(task, num_buffers, ctx);
  InferenceTask::fill_task_buffers(task, inputs_handles, outputs_handles);

  task->callback_func = InferenceTask::starpu_output_callback;
  ctx->self_keep_alive = ctx;
  task->callback_arg = ctx.get();

  assign_fixed_worker_if_needed(task);

  return task;
}

void
InferenceTask::allocate_task_buffers(
    starpu_task* task, size_t num_buffers,
    const std::shared_ptr<InferenceCallbackContext>& ctx)
{
  task->dyn_handles = static_cast<starpu_data_handle_t*>(
      malloc(num_buffers * sizeof(starpu_data_handle_t)));
  task->dyn_modes = static_cast<starpu_data_access_mode*>(
      malloc(num_buffers * sizeof(starpu_data_access_mode)));

  if (task->dyn_handles == nullptr || task->dyn_modes == nullptr) {
    starpu_task_destroy(task);
    cleanup(ctx);
    throw MemoryAllocationException("Failed to allocate task buffers.");
  }
}

void
InferenceTask::fill_task_buffers(
    starpu_task* task, const std::vector<starpu_data_handle_t>& inputs,
    const std::vector<starpu_data_handle_t>& outputs)
{
  const size_t num_inputs = inputs.size();
  const size_t num_buffers = num_inputs + outputs.size();

  for (size_t idx = 0; idx < num_inputs; ++idx) {
    task->dyn_handles[idx] = inputs[idx];
    task->dyn_modes[idx] = STARPU_R;
  }

  for (size_t idx = num_inputs; idx < num_buffers; ++idx) {
    task->dyn_handles[idx] = outputs[idx - num_inputs];
    task->dyn_modes[idx] = STARPU_W;
  }
}

void
InferenceTask::assign_fixed_worker_if_needed(starpu_task* task) const
{
  if (auto fixed_id = job_->get_fixed_worker_id(); fixed_id.has_value()) {
    int id = fixed_id.value();

    if (id < 0) {
      throw std::invalid_argument("Fixed worker ID must be non-negative");
    }

    task->workerid = static_cast<unsigned>(id);
    task->execute_on_a_specific_worker = 1;
  }
}

// =============================================================================
// StarPU Output Callback & Completion
// Finalize job upon output readiness. Release data handles and trigger user
// completion callbacks with measured latency.
// =============================================================================

void
InferenceTask::starpu_output_callback(void* arg)
{
  try {
    auto* ctx = static_cast<InferenceCallbackContext*>(arg);
    ctx->job->timing_info().callback_start_time =
        std::chrono::high_resolution_clock::now();

    ctx->remaining_outputs_to_acquire =
        static_cast<int>(ctx->outputs_handles.size());

    for (const auto& handle : ctx->outputs_handles) {
      InferenceTask::process_output_handle(handle, ctx);
    }
  }
  catch (const InferenceEngineException& e) {
    log_exception("starpu_output_callback", e);
  }
}

void
InferenceTask::acquire_output_handle(
    starpu_data_handle_t handle, InferenceCallbackContext* ctx)
{
  starpu_data_acquire_cb(
      handle, STARPU_R,
      [](void* cb_arg) {
        auto* cb_ctx = static_cast<InferenceCallbackContext*>(cb_arg);
        if (--cb_ctx->remaining_outputs_to_acquire == 0) {
          try {
            InferenceTask::finalize_inference_task(cb_ctx);
          }
          catch (const InferenceEngineException& e) {
            log_exception("starpu_output_callback", e);
          }
        }
      },
      ctx);
}

void
InferenceTask::process_output_handle(
    starpu_data_handle_t handle, InferenceCallbackContext* ctx)
{
  if (handle != nullptr) {
    InferenceTask::acquire_output_handle(handle, ctx);
  } else {
    if (--ctx->remaining_outputs_to_acquire == 0) {
      InferenceTask::finalize_inference_task(ctx);
    }
  }
}

void
InferenceTask::finalize_inference_task(void* arg)
{
  auto* ctx = static_cast<InferenceCallbackContext*>(arg);

  InferenceTask::release_output_data(ctx->outputs_handles);

  auto ctx_sptr =
      std::static_pointer_cast<InferenceCallbackContext>(ctx->self_keep_alive);

  InferenceTask::finalize_context(ctx_sptr);

  const auto end_time = std::chrono::high_resolution_clock::now();

  InferenceTask::record_and_run_completion_callback(ctx, end_time);

  ctx->self_keep_alive.reset();
}

void
InferenceTask::finalize_context(
    const std::shared_ptr<InferenceCallbackContext>& ctx_sptr)
{
  if (ctx_sptr) {
    InferenceTask::cleanup(ctx_sptr);
  }
}

void
InferenceTask::release_output_data(
    const std::vector<starpu_data_handle_t>& handles)
{
  for (const auto& handle : handles) {
    if (handle != nullptr) {
      starpu_data_release(handle);
    }
  }
}

void
InferenceTask::record_and_run_completion_callback(
    InferenceCallbackContext* ctx,
    std::chrono::high_resolution_clock::time_point end_time)
{
  if (!ctx->job) {
    return;
  }

  const double latency_ms = std::chrono::duration<double, std::milli>(
                                end_time - ctx->job->get_start_time())
                                .count();

  ctx->job->timing_info().callback_end_time = end_time;

  if (ctx->job->has_on_complete()) {
    const auto& callback = ctx->job->get_on_complete();
    callback(ctx->job->get_output_tensors(), latency_ms);
  }
}

// =============================================================================
// Exception Logging
// Catch and log known exceptions raised during job lifecycle.
// =============================================================================

void
InferenceTask::log_exception(
    const std::string& context, const std::exception& e)
{
  if (const auto* iee = dynamic_cast<const InferenceExecutionException*>(&e)) {
    log_error("InferenceExecutionException in " + context + ": " + iee->what());
  } else if (
      const auto* spe =
          dynamic_cast<const StarPUTaskSubmissionException*>(&e)) {
    log_error("StarPU submission error in " + context + ": " + spe->what());
  } else {
    log_error("std::exception in " + context + ": " + e.what());
  }
}

}  // namespace starpu_server
