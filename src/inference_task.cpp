#include "inference_task.hpp"

#include "exceptions.hpp"

struct InferenceCallbackContext {
  std::shared_ptr<InferenceJob> job;
  ProgramOptions opts;
  int id;
  std::vector<starpu_data_handle_t> input_handles;
  starpu_data_handle_t output_handle;
};

void
cleanup_inference_context(InferenceCallbackContext* ctx)
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
on_output_ready(void* arg)
{
  auto* ctx = static_cast<InferenceCallbackContext*>(arg);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - ctx->job->start_time)
                        .count();

  if (ctx->job->on_complete) {
    ctx->job->on_complete(ctx->job->output_tensor, latency_us);
  }

  cleanup_inference_context(ctx);
}

InferenceParams*
create_inference_params(
    const std::shared_ptr<InferenceJob>& job,
    torch::jit::script::Module* module)
{
  if (job->input_tensors.size() > InferLimits::MaxInputs) {
    throw InferenceExecutionException(
        "[ERROR] Too many input tensors, the maximum is : " +
        std::to_string(InferLimits::MaxInputs));
  }

  auto* inference_params = new InferenceParams();
  inference_params->num_inputs = job->input_tensors.size();
  inference_params->num_outputs = 1;
  inference_params->output_size = job->output_tensor.numel();
  inference_params->module = module;

  for (size_t i = 0; i < job->input_types.size(); ++i)
    inference_params->input_types[i] = job->input_types[i];

  for (size_t i = 0; i < job->input_tensors.size(); ++i) {
    const auto& tensor = job->input_tensors[i];
    if (tensor.dim() > InferLimits::MaxDims) {
      throw InferenceExecutionException(
          "[ERROR] Too many dimensions for input " + std::to_string(i));
    }
    inference_params->num_dims[i] = tensor.dim();
    for (int_fast64_t j = 0; j < tensor.dim(); ++j) {
      inference_params->dims[i][j] = tensor.size(j);
    }
  }

  return inference_params;
}

std::vector<starpu_data_handle_t>
register_input_handles(const std::vector<torch::Tensor>& input_tensors)
{
  std::vector<starpu_data_handle_t> handles(input_tensors.size());

  for (size_t i = 0; i < input_tensors.size(); ++i) {
    const auto& tensor = input_tensors[i];
    starpu_vector_data_register(
        &handles[i], STARPU_MAIN_RAM,
        reinterpret_cast<uintptr_t>(tensor.data_ptr()),
        static_cast<size_t>(tensor.numel()),
        static_cast<size_t>(tensor.element_size()));

    if (!handles[i]) {
      throw StarPURegistrationException(
          "[ERROR] Failed to register input handle at index " +
          std::to_string(i));
    }
  }

  return handles;
}

starpu_data_handle_t
register_output_handle(const torch::Tensor& output_tensor)
{
  starpu_data_handle_t handle = nullptr;
  starpu_vector_data_register(
      &handle, STARPU_MAIN_RAM,
      reinterpret_cast<uintptr_t>(output_tensor.data_ptr()),
      static_cast<size_t>(output_tensor.numel()),
      static_cast<size_t>(output_tensor.element_size()));

  if (!handle) {
    throw StarPURegistrationException(
        "[ERROR] Failed to register output handle.");
  }

  return handle;
}

void
log_exception(const std::string& context)
{
  try {
    throw;
  }
  catch (const InferenceExecutionException& e) {
    std::cerr << "[ERROR] InferenceExecutionException in " << context << ": "
              << e.what() << std::endl;
  }
  catch (const StarPUTaskSubmissionException& e) {
    std::cerr << "[ERROR] StarPU submission error in " << context << ": "
              << e.what() << std::endl;
  }
  catch (const std::exception& e) {
    std::cerr << "[ERROR] std::exception in " << context << ": " << e.what()
              << std::endl;
  }
  catch (...) {
    std::cerr << "[ERROR] Unknown exception in " << context << "." << std::endl;
  }
}

void
starpu_output_callback(void* arg)
{
  try {
    auto* ctx = static_cast<InferenceCallbackContext*>(arg);
    starpu_data_acquire_cb(
        ctx->output_handle, STARPU_R,
        [](void* cb_arg) {
          try {
            on_output_ready(cb_arg);
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
create_starpu_task(
    StarPUSetup& starpu, InferenceParams* inference_params,
    std::vector<starpu_data_handle_t> input_handles,
    starpu_data_handle_t output_handle, InferenceCallbackContext* ctx,
    const ProgramOptions& opts)
{
  size_t num_inputs = input_handles.size();
  size_t num_outputs = 1;
  size_t num_buffers = num_inputs + num_outputs;

  struct starpu_task* task = starpu_task_create();
  if (!task) {
    throw StarPUTaskCreationException("Failed to create StarPU task.");
  }
  task->nbuffers = static_cast<int>(num_buffers);
  task->cl = starpu.codelet();
  task->synchronous = opts.synchronous ? 1 : 0;
  task->cl_arg = inference_params;
  task->cl_arg_size = sizeof(InferenceParams);
  task->cl_arg_free = 1;
  task->destroy = 1;

  task->dyn_handles =
      (starpu_data_handle_t*)malloc(num_buffers * sizeof(*task->dyn_handles));
  task->dyn_modes =
      (starpu_data_access_mode*)malloc(num_buffers * sizeof(*task->dyn_modes));

  if (!task->dyn_handles || !task->dyn_modes) {
    free(task->dyn_handles);
    free(task->dyn_modes);
    starpu_task_destroy(task);
    throw MemoryAllocationException(
        "[ERROR] Memory allocation failed for dyn_handles or dyn_modes.");
  }

  for (size_t i = 0; i < num_inputs; ++i) {
    task->dyn_handles[i] = input_handles[i];
    task->dyn_modes[i] = STARPU_R;
  }

  task->dyn_handles[num_inputs] = output_handle;
  task->dyn_modes[num_inputs] = STARPU_W;

  task->callback_func = starpu_output_callback;
  task->callback_arg = ctx;

  return task;
}

void
submit_inference_task(
    StarPUSetup& starpu, std::shared_ptr<InferenceJob> job,
    torch::jit::script::Module& module, const ProgramOptions& opts)
{
  if (!job) {
    throw InvalidInferenceJobException("[ERROR] Job is null.");
  }

  size_t num_inputs = job->input_tensors.size();
  size_t num_outputs = 1;
  size_t num_buffers = num_outputs + num_inputs;

  auto input_handles = register_input_handles(job->input_tensors);
  auto output_handle = register_output_handle(job->output_tensor);

  InferenceParams* inference_params = create_inference_params(job, &module);

  auto* ctx = new InferenceCallbackContext{
      job, opts, job->job_id, input_handles, output_handle};

  struct starpu_task* task = create_starpu_task(
      starpu, inference_params, input_handles, output_handle, ctx, opts);

  int ret = starpu_task_submit(task);
  if (ret != 0) {
    cleanup_inference_context(ctx);
    throw StarPUTaskSubmissionException(
        "[ERROR] StarPU task submission failed (code " + std::to_string(ret) +
        ").");
  }
}