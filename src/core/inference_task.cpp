#include "inference_task.hpp"

#include <starpu.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <format>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "exceptions.hpp"
#include "inference_params.hpp"
#include "monitoring/metrics.hpp"
#include "output_slot_pool.hpp"
#include "starpu_setup.hpp"
#include "utils/device_type.hpp"
#include "utils/exception_logging.hpp"
#include "utils/tensor_validation.hpp"

namespace starpu_server {
namespace {

auto
resolve_dependencies(const InferenceTaskDependencies* deps)
    -> const InferenceTaskDependencies*
{
  return deps != nullptr ? deps : &kDefaultInferenceTaskDependencies;
}

auto
resolve_dependencies(
    const InferenceCallbackContext* ctx,
    const InferenceTaskDependencies* fallback)
    -> const InferenceTaskDependencies*
{
  if (ctx != nullptr) {
    if (ctx->dependencies != nullptr) {
      return ctx->dependencies;
    }
    if (ctx->dependencies_owner) {
      return ctx->dependencies_owner.get();
    }
  }
  return resolve_dependencies(fallback);
}

template <typename Fn>
auto
resolve_dependency_fn(Fn maybe_fn, Fn fallback) -> Fn
{
  return maybe_fn != nullptr ? maybe_fn : fallback;
}

auto
require_runtime_config(const RuntimeConfig* opts) -> const RuntimeConfig&
{
  if (opts == nullptr) {
    throw InferenceExecutionException("RuntimeConfig is null.");
  }
  return *opts;
}

enum class CallbackTerminalStatus {
  kSuccess,
  kFailure,
};

void
mark_callback_failure(
    InferenceCallbackContext* ctx, std::string_view reason,
    std::string_view message)
{
  if (ctx == nullptr || ctx->job == nullptr) {
    return;
  }

  if (!ctx->job->failure_info().has_value()) {
    const std::string model_name{ctx->job->model_name()};
    increment_inference_failure("callback", reason, model_name);

    InferenceJob::FailureInfo failure_info{};
    failure_info.stage = "callback";
    failure_info.reason = std::string(reason);
    failure_info.message = std::string(message);
    failure_info.metrics_reported = true;
    ctx->job->set_failure_info(std::move(failure_info));
  }

  // Force downstream async completion to enter the failure path.
  ctx->job->set_output_tensors({});
}

auto
resolve_terminal_status(const InferenceCallbackContext* ctx)
    -> CallbackTerminalStatus
{
  if (ctx != nullptr && ctx->job != nullptr &&
      ctx->job->failure_info().has_value()) {
    return CallbackTerminalStatus::kFailure;
  }
  return CallbackTerminalStatus::kSuccess;
}

void
finalize_or_fail_once(
    InferenceCallbackContext* ctx, CallbackTerminalStatus status,
    std::string_view context)
{
  if (ctx == nullptr) {
    log_error(std::format("{} received a null callback context", context));
    return;
  }

  if (status == CallbackTerminalStatus::kFailure && ctx->job != nullptr &&
      !ctx->job->failure_info().has_value()) {
    InferenceJob::FailureInfo failure_info{};
    failure_info.stage = "callback";
    failure_info.reason = "terminal_failure";
    failure_info.message =
        "Inference callback finalized after an unrecoverable error.";
    ctx->job->set_failure_info(std::move(failure_info));
    ctx->job->set_output_tensors({});
  }

  bool expected = false;
  if (!ctx->terminal_path_started.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    return;
  }

  try {
    InferenceTask::finalize_inference_task(ctx);
  }
  catch (const InferenceEngineException& exception) {
    InferenceTask::log_exception(std::string(context), exception);
  }
  catch (const std::exception& exception) {
    log_error(std::format(
        "std::exception in {} terminal path: {}", context, exception.what()));
  }
  catch (...) {
    log_error(std::format("Unknown exception in {} terminal path", context));
  }
}

void
decrement_remaining_and_finalize_if_done(
    InferenceCallbackContext* ctx, std::string_view context)
{
  if (ctx == nullptr) {
    return;
  }
  if (--ctx->remaining_outputs_to_acquire == 0) {
    finalize_or_fail_once(ctx, resolve_terminal_status(ctx), context);
  }
}

class StarpuHandleVectorGuard {
 public:
  explicit StarpuHandleVectorGuard(
      std::vector<starpu_data_handle_t>& handles) noexcept
      : handles_(handles)
  {
  }

  StarpuHandleVectorGuard(const StarpuHandleVectorGuard&) = delete;
  auto operator=(const StarpuHandleVectorGuard&) -> StarpuHandleVectorGuard& =
                                                        delete;
  StarpuHandleVectorGuard(StarpuHandleVectorGuard&&) = delete;
  auto operator=(StarpuHandleVectorGuard&&) -> StarpuHandleVectorGuard& =
                                                   delete;

  ~StarpuHandleVectorGuard() noexcept { reset(); }

  void dismiss() noexcept { active_ = false; }

 private:
  void reset() noexcept
  {
    if (!active_) {
      return;
    }
    for (auto* const handle : handles_) {
      if (handle != nullptr) {
        starpu_data_unregister(handle);
      }
    }
  }

  std::vector<starpu_data_handle_t>& handles_;
  bool active_ = true;
};

auto
register_tensor_handles(
    const std::vector<torch::Tensor>& tensors,
    std::string_view label_prefix) -> std::vector<starpu_data_handle_t>
{
  std::vector<starpu_data_handle_t> handles;
  handles.reserve(tensors.size());
  StarpuHandleVectorGuard unregister_guard(handles);

  for (size_t i = 0; i < tensors.size(); ++i) {
    handles.push_back(InferenceTask::safe_register_tensor_vector(
        tensors[i], std::format("{}[{}]", label_prefix, i)));
  }

  unregister_guard.dismiss();
  return handles;
}
}  // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
InferenceTaskDependencies kDefaultInferenceTaskDependencies{
    .dyn_handles_allocator = std::malloc,
    .dyn_handles_deallocator = std::free,
    .dyn_modes_allocator = std::malloc,
    .dyn_modes_deallocator = std::free,
    .task_create_fn = starpu_task_create,
    .starpu_data_acquire_fn = starpu_data_acquire_cb,
    .starpu_output_callback_hook = std::nullopt,
};

// =============================================================================
// Constructor
// =============================================================================

InferenceTask::InferenceTask(
    StarPUSetup* starpu, std::shared_ptr<InferenceJob> job,
    torch::jit::script::Module* model_cpu,
    std::vector<torch::jit::script::Module>* models_gpu,
    const RuntimeConfig* opts,
    const InferenceTaskDependencies& dependencies) noexcept
    : starpu_(starpu), job_(std::move(job)), model_cpu_(model_cpu),
      models_gpu_(models_gpu), opts_(opts),
      dependencies_(std::make_shared<InferenceTaskDependencies>(dependencies))
{
}

// =============================================================================
// InferenceCallbackContext
// Holds job metadata and StarPU handles needed during task lifecycle.
// Used as StarPU's callback context.
// =============================================================================

InferenceCallbackContext::InferenceCallbackContext(
    std::shared_ptr<InferenceJob> job_,
    std::shared_ptr<InferenceParams> params_,
    std::vector<starpu_data_handle_t> inputs_,
    std::vector<starpu_data_handle_t> outputs_) noexcept
    : job(std::move(job_)), inference_params(std::move(params_)),
      inputs_handles(std::move(inputs_)), outputs_handles(std::move(outputs_))
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
  if (auto error = tensor_validation::validate_cpu_contiguous_tensor(
          tensor, label, true)) {
    throw StarPURegistrationException(*error);
  }
  starpu_data_handle_t handle = nullptr;

  const auto numel = static_cast<uint64_t>(tensor.numel());
  const auto elem_size = static_cast<uint64_t>(tensor.element_size());
  const auto max_size = std::numeric_limits<size_t>::max();

  if (numel > max_size) {
    throw StarPURegistrationException(std::format(
        "Tensor '{}' has too many elements to fit in size_t", label));
  }
  if (elem_size > max_size) {
    throw StarPURegistrationException(std::format(
        "Tensor '{}' has an element size too large for size_t", label));
  }

  starpu_vector_data_register(
      &handle, STARPU_MAIN_RAM, std::bit_cast<uintptr_t>(tensor.data_ptr()),
      numel, elem_size);

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
  return register_tensor_handles(input_tensors, "input");
}

auto
InferenceTask::register_outputs_handles(
    const std::vector<torch::Tensor>& outputs_tensors)
    -> std::vector<starpu_data_handle_t>
{
  return register_tensor_handles(outputs_tensors, "output");
}

// =============================================================================
// Inference Parameters Setup
// Fill inference configuration structures passed to StarPU.
// =============================================================================

auto
InferenceTask::create_context(
    std::vector<starpu_data_handle_t> inputs,
    std::vector<starpu_data_handle_t> outputs)
    -> std::shared_ptr<InferenceCallbackContext>
{
  auto params = create_inference_params();
  auto ctx = std::make_shared<InferenceCallbackContext>(
      job_, std::move(params), std::move(inputs), std::move(outputs));
  ctx->dependencies_owner = dependencies_;
  ctx->dependencies = dependencies_.get();
  return ctx;
}

void
InferenceTask::check_limits(size_t num_inputs) const
{
  const auto& opts = require_runtime_config(opts_);
  if (num_inputs > opts.limits.max_inputs) {
    throw InferenceExecutionException(std::format(
        "Too many input tensors: max is {}", opts.limits.max_inputs));
  }

  if (models_gpu_->size() > opts.limits.max_models_gpu) {
    throw TooManyGpuModelsException(
        "Too many GPU models for the current configuration.");
  }
}

auto
InferenceTask::create_inference_params() -> std::shared_ptr<InferenceParams>
{
  const size_t num_inputs = job_->get_input_tensors().size();
  check_limits(num_inputs);

  const auto& opts = require_runtime_config(opts_);
  auto params = std::make_shared<InferenceParams>();

  params->limits.max_inputs = opts.limits.max_inputs;
  params->limits.max_dims = opts.limits.max_dims;

  fill_model_pointers(params);
  bind_runtime_job_info(params);
  fill_input_layout(params, num_inputs);

  params->num_inputs = num_inputs;
  params->num_outputs = job_->get_output_tensors().size();
  params->verbosity = opts.verbosity;

  return params;
}

void
InferenceTask::fill_model_pointers(
    const std::shared_ptr<InferenceParams>& params) const
{
  const auto& opts = require_runtime_config(opts_);
  params->models.model_cpu = model_cpu_;
  params->models.models_gpu.clear();
  params->models.device_ids.clear();

  if (opts.devices.ids.empty() || models_gpu_->empty()) {
    return;
  }

  const size_t replicas =
      std::min(models_gpu_->size(), opts.devices.ids.size());
  params->models.device_ids.reserve(replicas);
  params->models.models_gpu.reserve(replicas);
  for (size_t i = 0; i < replicas; ++i) {
    const int device_id = opts.devices.ids[i];
    if (device_id < 0) {
      continue;
    }
    params->models.device_ids.push_back(device_id);
    params->models.models_gpu.push_back(&(models_gpu_->at(i)));
  }
}

void
InferenceTask::bind_runtime_job_info(
    const std::shared_ptr<InferenceParams>& params) const
{
  params->request_id = job_->get_request_id();
  params->device.set_runtime_device_info =
      [job = job_](DeviceType executed_on, int device_id, int worker_id) {
        job->set_runtime_device_info(executed_on, device_id, worker_id);
      };
  params->timing.set_codelet_start_time =
      [job = job_](MonotonicClock::time_point started_at) {
        job->update_timing_info([started_at](detail::TimingInfo& timing) {
          timing.codelet_start_time = started_at;
        });
      };
  params->timing.set_codelet_end_time =
      [job = job_](MonotonicClock::time_point ended_at) {
        job->update_timing_info([ended_at](detail::TimingInfo& timing) {
          timing.codelet_end_time = ended_at;
        });
      };
  params->timing.set_inference_start_time =
      [job = job_](MonotonicClock::time_point started_at) {
        job->update_timing_info([started_at](detail::TimingInfo& timing) {
          timing.inference_start_time = started_at;
        });
      };
}

void
InferenceTask::fill_input_layout(
    const std::shared_ptr<InferenceParams>& params, size_t num_inputs) const
{
  const auto& opts = require_runtime_config(opts_);
  params->layout.input_types.clear();
  params->layout.input_types.reserve(num_inputs);
  if (const auto& job_types = job_->get_input_types();
      job_types.size() >= num_inputs) {
    params->layout.input_types.insert(
        params->layout.input_types.end(), job_types.begin(),
        job_types.begin() + static_cast<std::ptrdiff_t>(num_inputs));
  } else {
    const auto& inputs = job_->get_input_tensors();
    for (size_t i = 0; i < num_inputs; ++i) {
      const auto& tensor = inputs[i];
      if (!tensor.defined()) {
        throw InferenceExecutionException(
            "Input tensor is undefined; cannot infer input type.");
      }
      params->layout.input_types.push_back(tensor.scalar_type());
    }
  }

  params->layout.num_dims.resize(num_inputs);
  params->layout.dims.resize(num_inputs);

  for (size_t i = 0; i < num_inputs; ++i) {
    auto dims_from_config = [i, &opts]() -> std::vector<int64_t> {
      if (!opts.model.has_value() || i >= opts.model->inputs.size()) {
        return {};
      }
      return opts.model->inputs[i].dims;
    };

    auto dims_from_tensor = [this, i, &opts]() -> std::vector<int64_t> {
      const auto& tensor = job_->get_input_tensors()[i];
      const int64_t dim = tensor.dim();
      if (dim > static_cast<int64_t>(opts.limits.max_dims)) {
        throw InferenceExecutionException(std::format(
            "Input tensor has too many dimensions: max is {}",
            opts.limits.max_dims));
      }
      const auto sizes = tensor.sizes();
      const auto first_dims = std::views::take(sizes, dim);
      return {first_dims.begin(), first_dims.end()};
    };

    std::vector<int64_t> dims = dims_from_config();
    const bool used_config_dims = !dims.empty();
    if (dims.empty()) {
      dims = dims_from_tensor();
    }

    if (!dims.empty()) {
      if (const auto effective = job_->effective_batch_size();
          effective.has_value()) {
        dims.front() = std::max<int64_t>(1, *effective);
      } else if (used_config_dims) {
        dims.front() = std::max<int64_t>(1, dims.front());
      }
    }

    if (dims.size() > opts.limits.max_dims) {
      throw InferenceExecutionException(std::format(
          "Input tensor has too many dimensions: max is {}",
          opts.limits.max_dims));
    }

    params->layout.num_dims[i] = static_cast<int>(dims.size());
    params->layout.dims[i] = std::move(dims);
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

  if (!ctx->keep_input_handles) {
    for (auto& handle : ctx->inputs_handles) {
      if (handle != nullptr) {
        starpu_data_unregister_submit(handle);
        handle = nullptr;
      }
    }
  }

  if (!ctx->keep_output_handles) {
    for (auto& handle : ctx->outputs_handles) {
      if (handle != nullptr) {
        starpu_data_unregister_submit(handle);
        handle = nullptr;
      }
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
  auto ctx =
      create_context(std::move(inputs_handles), std::move(outputs_handles));
  starpu_task* task =
      create_task(ctx->inputs_handles, ctx->outputs_handles, ctx);

  const auto submitted_at = MonotonicClock::now();
  job_->update_timing_info([submitted_at](detail::TimingInfo& timing) {
    timing.before_starpu_submitted_time = submitted_at;
  });

  const int ret = starpu_task_submit(task);
  if (ret != 0) {
    cleanup(ctx);
    if (ctx != nullptr) {
      ctx->self_keep_alive.reset();
    }
    if (task != nullptr) {
      starpu_task_destroy(task);
    }
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
  const auto& opts = require_runtime_config(opts_);
  const size_t num_inputs = inputs_handles.size();
  const size_t num_buffers = num_inputs + outputs_handles.size();

  const auto* dependencies = resolve_dependencies(dependencies_.get());
  const auto task_create = resolve_dependency_fn(
      dependencies->task_create_fn,
      kDefaultInferenceTaskDependencies.task_create_fn);
  auto* task = task_create != nullptr ? task_create() : nullptr;
  if (task == nullptr) {
    throw StarPUTaskCreationException("Failed to create StarPU task.");
  }

  task->nbuffers = static_cast<int>(num_buffers);
  // Some unit tests build tasks without a real StarPUSetup instance because
  // they only validate cleanup/dependency behavior, not task execution.
  task->cl = (starpu_ != nullptr) ? starpu_->get_codelet() : nullptr;
  task->synchronous = opts.batching.synchronous ? 1 : 0;
  task->cl_arg = ctx->inference_params.get();
  int min_priority = STARPU_DEFAULT_PRIO;
  int max_priority = STARPU_DEFAULT_PRIO;
  // Some unit tests call create_task() without bootstrapping StarPU. In that
  // case, querying scheduler bounds segfaults, so we keep default priority.
  if (starpu_is_initialized() > 0) {
    min_priority = starpu_sched_get_min_priority();
    max_priority = starpu_sched_get_max_priority();
  }
  task->priority =
      std::max(min_priority, max_priority - ctx->job->get_request_id());
  task->destroy = 1;

  if (ctx != nullptr && ctx->dependencies == nullptr) {
    if (ctx->dependencies_owner) {
      ctx->dependencies = ctx->dependencies_owner.get();
    } else {
      ctx->dependencies_owner = dependencies_;
      ctx->dependencies = dependencies_.get();
    }
  }

  try {
    assign_fixed_worker_if_needed(task);
  }
  catch (...) {
    starpu_task_destroy(task);
    cleanup(ctx);
    throw;
  }

  InferenceTask::allocate_task_buffers(task, num_buffers, ctx);
  InferenceTask::fill_task_buffers(task, inputs_handles, outputs_handles);

  task->callback_func = InferenceTask::starpu_output_callback;
  ctx->self_keep_alive = ctx;
  task->callback_arg = ctx.get();

  return task;
}

void
InferenceTask::allocate_task_buffers(
    starpu_task* task, size_t num_buffers,
    const std::shared_ptr<InferenceCallbackContext>& ctx)
{
  const auto* dependencies = resolve_dependencies(ctx.get(), nullptr);
  const auto handles_allocator = resolve_dependency_fn(
      dependencies->dyn_handles_allocator,
      kDefaultInferenceTaskDependencies.dyn_handles_allocator);
  const auto handles_deallocator = resolve_dependency_fn(
      dependencies->dyn_handles_deallocator,
      kDefaultInferenceTaskDependencies.dyn_handles_deallocator);
  const auto modes_allocator = resolve_dependency_fn(
      dependencies->dyn_modes_allocator,
      kDefaultInferenceTaskDependencies.dyn_modes_allocator);
  const auto modes_deallocator = resolve_dependency_fn(
      dependencies->dyn_modes_deallocator,
      kDefaultInferenceTaskDependencies.dyn_modes_deallocator);

  auto handles_owner = std::unique_ptr<void, void (*)(void*)>(
      handles_allocator(num_buffers * sizeof(starpu_data_handle_t)),
      handles_deallocator);
  if (!handles_owner) {
    task->dyn_handles = nullptr;
    task->dyn_modes = nullptr;
    starpu_task_destroy(task);
    cleanup(ctx);
    throw MemoryAllocationException("Failed to allocate task buffers.");
  }

  auto modes_owner = std::unique_ptr<void, void (*)(void*)>(
      modes_allocator(num_buffers * sizeof(starpu_data_access_mode)),
      modes_deallocator);
  if (!modes_owner) {
    task->dyn_handles = nullptr;
    task->dyn_modes = nullptr;
    starpu_task_destroy(task);
    cleanup(ctx);
    throw MemoryAllocationException("Failed to allocate task buffers.");
  }

  task->dyn_handles =
      static_cast<starpu_data_handle_t*>(handles_owner.release());
  task->dyn_modes =
      static_cast<starpu_data_access_mode*>(modes_owner.release());
}

void
InferenceTask::fill_task_buffers(
    starpu_task* task, const std::vector<starpu_data_handle_t>& inputs,
    const std::vector<starpu_data_handle_t>& outputs)
{
  const size_t num_inputs = inputs.size();
  const size_t num_buffers = num_inputs + outputs.size();

  std::span<starpu_data_handle_t> handles(task->dyn_handles, num_buffers);
  std::span<starpu_data_access_mode> modes(task->dyn_modes, num_buffers);

  for (size_t idx = 0; idx < num_inputs; ++idx) {
    handles[idx] = inputs[idx];
    modes[idx] = STARPU_R;
  }

  for (size_t idx = 0; idx < outputs.size(); ++idx) {
    handles[num_inputs + idx] = outputs[idx];
    modes[num_inputs + idx] = STARPU_W;
  }
}

void
InferenceTask::assign_fixed_worker_if_needed(starpu_task* task) const
{
  if (auto fixed_id = job_->get_fixed_worker_id(); fixed_id.has_value()) {
    int worker_id = fixed_id.value();

    if (worker_id < 0) {
      throw std::invalid_argument("Fixed worker ID must be non-negative");
    }

    const auto total_workers = static_cast<int>(starpu_worker_get_count());
    if (worker_id >= total_workers) {
      throw std::out_of_range("Fixed worker ID exceeds available workers");
    }

    task->workerid = static_cast<unsigned>(worker_id);
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
  auto* ctx = static_cast<InferenceCallbackContext*>(arg);
  if (ctx == nullptr) {
    log_error("starpu_output_callback received null context");
    return;
  }

  const auto callback_start = MonotonicClock::now();
  if (ctx->job != nullptr) {
    ctx->job->update_timing_info([callback_start](detail::TimingInfo& timing) {
      timing.callback_start_time = callback_start;
    });
  }

  ctx->remaining_outputs_to_acquire =
      static_cast<int>(ctx->outputs_handles.size());
  ctx->outputs_handles_to_release = ctx->outputs_handles;
  if (ctx->outputs_handles.empty()) {
    finalize_or_fail_once(
        ctx, resolve_terminal_status(ctx), "starpu_output_callback");
    return;
  }

  const auto* dependencies = resolve_dependencies(ctx, nullptr);
  bool bypass_remaining_handles = false;

  for (std::size_t index = 0; index < ctx->outputs_handles.size(); ++index) {
    const auto handle = ctx->outputs_handles[index];
    if (bypass_remaining_handles) {
      ctx->outputs_handles_to_release[index] = nullptr;
      decrement_remaining_and_finalize_if_done(ctx, "starpu_output_callback");
      continue;
    }

    try {
      if (dependencies->starpu_output_callback_hook.has_value()) {
        const auto& hook = *dependencies->starpu_output_callback_hook;
        hook(ctx);
      }
      InferenceTask::process_output_handle(handle, ctx);
    }
    catch (const InferenceEngineException& exception) {
      log_exception("starpu_output_callback", exception);
      mark_callback_failure(ctx, "output_acquire_failed", exception.what());
      ctx->outputs_handles_to_release[index] = nullptr;
      bypass_remaining_handles = true;
      decrement_remaining_and_finalize_if_done(ctx, "starpu_output_callback");
    }
    catch (const std::exception& exception) {
      log_error(std::format(
          "std::exception in starpu_output_callback: {}", exception.what()));
      mark_callback_failure(ctx, "output_callback_exception", exception.what());
      ctx->outputs_handles_to_release[index] = nullptr;
      bypass_remaining_handles = true;
      decrement_remaining_and_finalize_if_done(ctx, "starpu_output_callback");
    }
    catch (...) {
      log_error("Unknown exception in starpu_output_callback");
      mark_callback_failure(
          ctx, "output_callback_unknown_exception",
          "Unknown non-standard exception in output callback.");
      ctx->outputs_handles_to_release[index] = nullptr;
      bypass_remaining_handles = true;
      decrement_remaining_and_finalize_if_done(ctx, "starpu_output_callback");
    }
  }
}

void
InferenceTask::acquire_output_handle(
    starpu_data_handle_t handle, InferenceCallbackContext* ctx)
{
  const auto* dependencies = resolve_dependencies(ctx, nullptr);
  const auto data_acquire_fn = resolve_dependency_fn(
      dependencies->starpu_data_acquire_fn,
      kDefaultInferenceTaskDependencies.starpu_data_acquire_fn);

  if (data_acquire_fn == nullptr) {
    log_error("starpu_data_acquire_fn is null; cannot acquire output handle.");
    throw StarPURegistrationException("StarPU data acquire callback is null.");
  }

  const int ret = data_acquire_fn(
      handle, STARPU_R,
      [](void* cb_arg) {
        auto* cb_ctx = static_cast<InferenceCallbackContext*>(cb_arg);
        decrement_remaining_and_finalize_if_done(
            cb_ctx, "starpu_output_callback");
      },
      ctx);

  if (ret < 0) {
    log_error(std::format("starpu_data_acquire_cb failed with code {}", ret));
    throw StarPURegistrationException(
        "Failed to acquire output data handle from StarPU.");
  }
}

void
InferenceTask::process_output_handle(
    starpu_data_handle_t handle, InferenceCallbackContext* ctx)
{
  if (handle != nullptr) {
    InferenceTask::acquire_output_handle(handle, ctx);
  } else {
    decrement_remaining_and_finalize_if_done(ctx, "starpu_output_callback");
  }
}

namespace {
void
copy_outputs_from_pool(InferenceCallbackContext* ctx)
{
  const auto copy_start = MonotonicClock::now();
  std::size_t total_bytes = 0;
  const auto& base_ptrs = ctx->output_pool->base_ptrs(ctx->output_slot_id);
  const auto& job_outs = ctx->job->get_output_tensors();
  const size_t tensor_count = std::min(base_ptrs.size(), job_outs.size());
  for (size_t i = 0; i < tensor_count; ++i) {
    const auto& job_output_tensor = job_outs[i];
    if (!job_output_tensor.defined() || !job_output_tensor.is_cpu() ||
        !job_output_tensor.is_contiguous()) {
      throw InvalidInferenceJobException(
          "Job output tensor must be defined, CPU and contiguous");
    }
    std::memcpy(
        job_output_tensor.data_ptr(), base_ptrs[i], job_output_tensor.nbytes());
    total_bytes += job_output_tensor.nbytes();
  }
  const auto copy_end = MonotonicClock::now();
  const double copy_ms =
      std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
  const auto worker_type_label =
      std::string_view(to_string(ctx->job->get_executed_on()));
  observe_io_copy_latency(
      "d2h", ctx->job->get_worker_id(), ctx->job->get_device_id(),
      worker_type_label, copy_ms);
  increment_transfer_bytes(
      "d2h", ctx->job->get_worker_id(), ctx->job->get_device_id(),
      worker_type_label, total_bytes);
}
}  // namespace

void
InferenceTask::finalize_inference_task(void* arg)
{
  auto* ctx = static_cast<InferenceCallbackContext*>(arg);

  if (ctx->output_pool != nullptr && ctx->output_slot_id >= 0 && ctx->job) {
    run_with_logged_exceptions(
        [ctx]() { copy_outputs_from_pool(ctx); },
        ExceptionLoggingMessages{
            "Output copy from pool failed: ",
            "Output copy from pool failed due to an unknown exception."});
  }

  const auto& handles_to_release = ctx->outputs_handles_to_release.empty()
                                       ? ctx->outputs_handles
                                       : ctx->outputs_handles_to_release;
  InferenceTask::release_output_data(handles_to_release);

  auto ctx_sptr =
      std::static_pointer_cast<InferenceCallbackContext>(ctx->self_keep_alive);

  // Notify (release pooled slot etc.) before we clean up
  if (ctx->on_finished) {
    run_with_logged_exceptions(
        [ctx]() { ctx->on_finished(); },
        ExceptionLoggingMessages{
            "Exception in on_finished: ",
            "Unknown exception in on_finished callback"});
  }

  InferenceTask::finalize_context(ctx_sptr);

  const auto end_time = MonotonicClock::now();

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
    InferenceCallbackContext* ctx, MonotonicClock::time_point end_time)
{
  if (!ctx->job) {
    return;
  }

  const double latency_ms = std::chrono::duration<double, std::milli>(
                                end_time - ctx->job->get_start_time())
                                .count();

  ctx->job->update_timing_info([end_time](detail::TimingInfo& timing) {
    timing.callback_end_time = end_time;
  });

  auto callback = ctx->job->take_on_complete();
  if (callback) {
    run_with_logged_exceptions(
        [ctx, callback = std::move(callback), latency_ms]() mutable {
          callback(ctx->job->get_output_tensors(), latency_ms);
        },
        ExceptionLoggingMessages{
            "Exception in completion callback: ",
            "Unknown exception in completion callback"});
  }

  ctx->job->release_input_memory_holders();
}

// =============================================================================
// Exception Logging
// Catch and log known exceptions raised during job lifecycle.
// =============================================================================

void
InferenceTask::log_exception(
    const std::string& context, const std::exception& exception)
{
  if (const auto* iee =
          dynamic_cast<const InferenceExecutionException*>(&exception)) {
    log_error("InferenceExecutionException in " + context + ": " + iee->what());
  } else if (
      const auto* spe =
          dynamic_cast<const StarPUTaskSubmissionException*>(&exception)) {
    log_error("StarPU submission error in " + context + ": " + spe->what());
  } else {
    log_error("std::exception in " + context + ": " + exception.what());
  }
}

}  // namespace starpu_server
