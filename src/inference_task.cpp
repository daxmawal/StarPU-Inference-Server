#include "inference_task.hpp"

#include <chrono>

#include "exceptions.hpp"
#include "starpu_setup.hpp"

InferenceTask::InferenceTask(
    StarPUSetup& starpu, std::shared_ptr<InferenceJob> job,
    torch::jit::script::Module& modele_cpu,
    torch::jit::script::Module& modele_gpu, const ProgramOptions& opts)
    : starpu_(starpu), job_(std::move(job)), modele_cpu_(modele_cpu),
      modele_gpu_(modele_gpu), opts_(opts)
{
}

void
InferenceTask::cleanup(InferenceCallbackContext* ctx)
{
  if (ctx) {
    for (size_t i = 0; i < ctx->input_handles.size(); ++i) {
      if (ctx->input_handles[i]) {
        starpu_data_unregister_submit(ctx->input_handles[i]);
      }
    }
    if (ctx->output_handle) {
      starpu_data_unregister_submit(ctx->output_handle);
    }
    delete ctx;
  }
}

void
InferenceTask::on_output_ready_and_cleanup(void* arg)
{
  auto* ctx = static_cast<InferenceCallbackContext*>(arg);
  const float* output_ptr =
      static_cast<float*>(starpu_data_get_user_data(ctx->output_handle));
  starpu_data_release(ctx->output_handle);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto latency_ms =
      std::chrono::duration<double, std::milli>(end_time - ctx->job->start_time)
          .count();

  ctx->job->timing_info.callback_end_time =
      std::chrono::high_resolution_clock::now();

  if (ctx->job->on_complete) {
    ctx->job->on_complete(ctx->job->output_tensor, latency_ms);
  }

  cleanup(ctx);
}

std::shared_ptr<InferenceParams>
InferenceTask::create_inference_params()
{
  if (job_->input_tensors.size() > InferLimits::MaxInputs) {
    throw InferenceExecutionException(
        "[ERROR] Too many input tensors, the maximum is: " +
        std::to_string(InferLimits::MaxInputs));
  }

  auto inference_params = std::make_shared<InferenceParams>();
  inference_params->modele_cpu = &modele_cpu_;
  inference_params->modele_gpu = &modele_gpu_;
  inference_params->num_inputs = job_->input_tensors.size();
  inference_params->num_outputs = 1;
  inference_params->output_size = job_->output_tensor.numel();
  inference_params->job_id = job_->job_id;
  inference_params->executed_on = &job_->executed_on;
  inference_params->device_id = &job_->device_id;
  inference_params->codelet_start_time =
      &(job_->timing_info.codelet_start_time);
  inference_params->codelet_end_time = &(job_->timing_info.codelet_end_time);
  inference_params->inference_start_time =
      &(job_->timing_info.inference_start_time);

  auto offset = static_cast<std::ptrdiff_t>(inference_params->num_inputs);
  std::copy(
      job_->input_types.begin(), job_->input_types.begin() + offset,
      inference_params->input_types.begin());

  for (size_t i = 0; i < inference_params->num_inputs; ++i) {
    const auto& tensor = job_->input_tensors[i];
    auto dim = tensor.dim();
    inference_params->num_dims[i] = dim;
    std::copy_n(tensor.sizes().data(), dim, inference_params->dims[i].begin());
  }

  inference_params->verbosity = opts_.verbosity;

  return inference_params;
}

starpu_data_handle_t
InferenceTask::safe_register_tensor_vector(
    const torch::Tensor& tensor, const std::string& label)
{
  if (!tensor.defined() || !tensor.data_ptr()) {
    throw StarPURegistrationException(
        "[ERROR] Tensor '" + label + "' is undefined or has no data pointer.");
  }

  starpu_data_handle_t handle = nullptr;

  starpu_vector_data_register(
      &handle, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(tensor.data_ptr()),
      static_cast<size_t>(tensor.numel()),
      static_cast<size_t>(tensor.element_size()));

  if (!handle) {
    throw StarPURegistrationException(
        "[ERROR] Failed to register tensor '" + label + "' with StarPU.");
  }

  return handle;
}

std::vector<starpu_data_handle_t>
InferenceTask::register_input_handles(
    const std::vector<torch::Tensor>& input_tensors)
{
  std::vector<starpu_data_handle_t> handles;
  handles.reserve(input_tensors.size());

  for (size_t i = 0; i < input_tensors.size(); ++i) {
    handles.push_back(safe_register_tensor_vector(
        input_tensors[i], "input[" + std::to_string(i) + "]"));
  }

  return handles;
}

starpu_data_handle_t
InferenceTask::register_output_handle(const torch::Tensor& output_tensor)
{
  auto output_handle = safe_register_tensor_vector(output_tensor, "output");
  starpu_data_set_user_data(output_handle, output_tensor.data_ptr());
  return output_handle;
}

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

void
InferenceTask::starpu_output_callback(void* arg)
{
  try {
    auto* ctx = static_cast<InferenceCallbackContext*>(arg);
    ctx->job->timing_info.callback_start_time =
        std::chrono::high_resolution_clock::now();
    starpu_data_acquire_cb(
        ctx->output_handle, STARPU_R,
        [](void* cb_arg) {
          try {
            InferenceTask::on_output_ready_and_cleanup(cb_arg);
          }
          catch (...) {
            log_exception("on_output_ready");
          }
        },
        ctx);
  }
  catch (...) {
    log_exception("starpu_output_callback");
  }
}

starpu_task*
InferenceTask::create_task(
    const std::vector<starpu_data_handle_t>& input_handles,
    const starpu_data_handle_t& output_handle, InferenceCallbackContext* ctx)
{
  size_t num_inputs = input_handles.size();
  size_t num_outputs = 1;
  size_t num_buffers = num_inputs + num_outputs;

  struct starpu_task* task = starpu_task_create();
  if (!task) {
    throw StarPUTaskCreationException("Failed to create StarPU task.");
  }
  task->nbuffers = static_cast<int>(num_buffers);
  task->cl = starpu_.codelet();
  task->synchronous = opts_.synchronous ? 1 : 0;
  task->cl_arg = ctx->inference_params.get();
  task->cl_arg_size = sizeof(InferenceParams);
  task->priority = STARPU_MAX_PRIO - ctx->job->job_id;
  task->dyn_handles = static_cast<starpu_data_handle_t*>(
      malloc(num_buffers * sizeof(starpu_data_handle_t)));
  task->dyn_modes = static_cast<starpu_data_access_mode*>(
      malloc(num_buffers * sizeof(starpu_data_access_mode)));

  if (!task->dyn_handles || !task->dyn_modes) {
    if (task->dyn_handles)
      free(task->dyn_handles);
    if (task->dyn_modes)
      free(task->dyn_modes);
    starpu_task_destroy(task);
    cleanup(ctx);
    throw MemoryAllocationException(
        "Memory allocation failed for task buffers.");
  }

  for (size_t i = 0; i < num_inputs; ++i) {
    task->dyn_handles[i] = input_handles[i];
    task->dyn_modes[i] = STARPU_R;
  }

  task->dyn_handles[num_inputs] = output_handle;
  task->dyn_modes[num_inputs] = STARPU_W;

  task->callback_func = InferenceTask::starpu_output_callback;
  task->callback_arg = ctx;

  if (job_->fixed_worker_id.has_value()) {
    task->execute_on_a_specific_worker = 1;
    task->workerid = job_->fixed_worker_id.value();
  }

  return task;
}

void
InferenceTask::submit()
{
  if (!job_) {
    throw InvalidInferenceJobException("[ERROR] Job is null.");
  }

  auto input_handles = register_input_handles(job_->input_tensors);
  auto output_handle = register_output_handle(job_->output_tensor);

  std::shared_ptr<InferenceParams> inference_params = create_inference_params();

  auto* ctx = new InferenceCallbackContext{job_,          inference_params,
                                           opts_,         job_->job_id,
                                           input_handles, output_handle};

  struct starpu_task* task = create_task(input_handles, output_handle, ctx);

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