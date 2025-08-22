#include "inference_runner.hpp"

#include <ATen/core/ScalarType.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <format>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "client_utils.hpp"
#include "exceptions.hpp"
#include "inference_queue.hpp"
#include "inference_validator.hpp"
#include "input_generator.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"
#include "starpu_task_worker.hpp"
#include "warmup.hpp"

namespace starpu_server {
constexpr size_t NUM_PREGENERATED_INPUTS = 10;
constexpr int NUM_WARMUP_ITERATIONS = 2;

// =============================================================================
// InferenceJob: Encapsulates a single inference task, including input data,
// types, ID, and completion callback
// =============================================================================

InferenceJob::InferenceJob(
    std::vector<torch::Tensor> inputs, std::vector<at::ScalarType> types,
    int job_identifier,
    std::function<void(std::vector<torch::Tensor>, double)> callback)
    : input_tensors_(std::move(inputs)), input_types_(std::move(types)),
      job_id_(job_identifier), on_complete_(std::move(callback)),
      start_time_(std::chrono::high_resolution_clock::now())
{
}

auto
InferenceJob::make_shutdown_job() -> std::shared_ptr<InferenceJob>
{
  auto job = std::make_shared<InferenceJob>();
  job->is_shutdown_signal_ = true;
  return job;
}

// =============================================================================
// Client Logic: Generates and enqueues inference jobs into the shared queue
// =============================================================================

static void
client_worker(
    InferenceQueue& queue, const RuntimeConfig& opts,
    const std::vector<torch::Tensor>& outputs_ref, const int iterations)
{
  auto pregen_inputs =
      std::make_unique<std::vector<std::vector<torch::Tensor>>>(
          client_utils::pre_generate_inputs(opts, NUM_PREGENERATED_INPUTS));
  // RNG for synthetic test data only; not used for security.  // NOSONAR
  thread_local std::mt19937 rng(std::random_device{}());

  auto next_time = std::chrono::steady_clock::now();
  const auto delay = std::chrono::milliseconds(opts.delay_ms);
  for (auto job_id = 0; job_id < iterations; ++job_id) {
    std::this_thread::sleep_until(next_time);
    next_time += delay;
    const auto& inputs = client_utils::pick_random_input(*pregen_inputs, rng);
    auto job = client_utils::create_job(inputs, outputs_ref, job_id);
    client_utils::log_job_enqueued(
        opts, job_id, iterations, job->timing_info().enqueued_time);
    queue.push(job);
  }

  queue.shutdown();
}

// =============================================================================
// Model Loading and Cloning to GPU
// =============================================================================

static auto
load_model(const std::string& model_path) -> torch::jit::script::Module
{
  try {
    return torch::jit::load(model_path);
  }
  catch (const c10::Error& e) {
    log_error("Failed to load model: " + std::string(e.what()));
    throw;
  }
}

static auto
clone_model_to_gpus(
    const torch::jit::script::Module& model_cpu,
    const std::vector<int>& device_ids)
    -> std::vector<torch::jit::script::Module>
{
  std::vector<torch::jit::script::Module> models_gpu;
  models_gpu.reserve(device_ids.size());

  for (const auto& device_id : device_ids) {
    torch::jit::script::Module model_gpu = model_cpu.clone();
    model_gpu.to(
        torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(device_id)));
    models_gpu.emplace_back(std::move(model_gpu));
  }

  return models_gpu;
}

// =============================================================================
// Input Generation and Reference Inference Execution (CPU only)
// =============================================================================

static auto
generate_inputs(const std::vector<TensorConfig>& tensors)
    -> std::vector<torch::Tensor>
{
  return input_generator::generate_random_inputs(tensors);
}

static auto
run_reference_inference(
    torch::jit::script::Module& model,
    const std::vector<torch::Tensor>& inputs) -> std::vector<torch::Tensor>
{
  std::vector<torch::Tensor> output_refs;
  const std::vector<torch::IValue> input_ivalues(inputs.begin(), inputs.end());

  if (const auto output = model.forward(input_ivalues); output.isTensor()) {
    output_refs.push_back(output.toTensor());
  } else if (output.isTuple()) {
    for (const auto& val : output.toTuple()->elements()) {
      if (val.isTensor()) {
        output_refs.push_back(val.toTensor());
      }
    }
  } else if (output.isTensorList()) {
    output_refs.insert(
        output_refs.end(), output.toTensorList().begin(),
        output.toTensorList().end());
  } else {
    log_error("Unsupported output type from model.");
    throw UnsupportedModelOutputTypeException("Unsupported model output type");
  }

  return output_refs;
}

// =============================================================================
// Model and Reference Output Loader: returns CPU model, GPU clones, and ref
// outputs
// =============================================================================

auto
load_model_and_reference_output(const RuntimeConfig& opts)
    -> std::tuple<
        torch::jit::script::Module, std::vector<torch::jit::script::Module>,
        std::vector<torch::Tensor>>
{
  try {
    auto model_cpu = load_model(opts.model_path);
    auto models_gpu = opts.use_cuda
                          ? clone_model_to_gpus(model_cpu, opts.device_ids)
                          : std::vector<torch::jit::script::Module>{};
    auto inputs = generate_inputs(opts.inputs);
    auto output_refs = run_reference_inference(model_cpu, inputs);

    return {model_cpu, models_gpu, output_refs};
  }
  catch (const c10::Error& e) {
    log_error(
        "Failed to load model or run reference inference: " +
        std::string(e.what()));
    return {};
  }
}

// =============================================================================
// Warmup Phase: Run small number of inference tasks to warm up StarPU & Torch
// =============================================================================

void
run_warmup(
    const RuntimeConfig& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<torch::Tensor>& outputs_ref)
{
  if (!opts.use_cuda) {
    return;
  }

  log_info(
      opts.verbosity,
      std::format(
          "Starting warmup with {} iterations per CUDA device...",
          NUM_WARMUP_ITERATIONS));

  WarmupRunner warmup_runner(opts, starpu, model_cpu, models_gpu, outputs_ref);
  warmup_runner.run(NUM_WARMUP_ITERATIONS);

  log_info(opts.verbosity, "Warmup complete. Proceeding to real inference.\n");
}

// =============================================================================
// Result Processing: Print latency breakdowns and validate results
// =============================================================================

static void
process_results(
    const std::vector<InferenceResult>& results,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    VerbosityLevel verbosity)
{
  for (const auto& result : results) {
    if (result.results.empty() || !result.results[0].defined()) {
      log_error(std::format("[Client] Job {} failed.", result.job_id));
      continue;
    }

    torch::jit::script::Module* cpu_model = &model_cpu;
    if (result.executed_on == DeviceType::CUDA) {
      const auto device_id = static_cast<size_t>(result.device_id);
      if (device_id < models_gpu.size()) {
        cpu_model = &models_gpu[device_id];
      }
    }

    validate_inference_result(result, *cpu_model, verbosity);
  }
}

// =============================================================================
// Main Inference Loop: Initializes models, runs warmup, starts client/server,
// waits for all jobs to complete, processes results
// =============================================================================

void
run_inference_loop(const RuntimeConfig& opts, StarPUSetup& starpu)
{
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> outputs_ref;

  std::tie(model_cpu, models_gpu, outputs_ref) =
      load_model_and_reference_output(opts);

  run_warmup(opts, starpu, model_cpu, models_gpu, outputs_ref);

  InferenceQueue queue;
  std::vector<InferenceResult> results;
  std::mutex results_mutex;

  if (opts.iterations > 0) {
    results.reserve(static_cast<size_t>(opts.iterations));
  }

  std::atomic completed_jobs = 0;
  std::condition_variable all_done_cv;
  std::mutex all_done_mutex;

  StarPUTaskRunnerConfig config{};
  config.queue = &queue;
  config.model_cpu = &model_cpu;
  config.models_gpu = &models_gpu;
  config.starpu = &starpu;
  config.opts = &opts;
  config.results = &results;
  config.results_mutex = &results_mutex;
  config.completed_jobs = &completed_jobs;
  config.all_done_cv = &all_done_cv;
  StarPUTaskRunner worker(config);

  const std::jthread server(&StarPUTaskRunner::run, &worker);
  const std::jthread client([&queue, &opts, &outputs_ref]() {
    client_worker(queue, opts, outputs_ref, opts.iterations);
  });

  {
    std::unique_lock lock(all_done_mutex);
    all_done_cv.wait(lock, [&completed_jobs, &opts]() {
      return completed_jobs.load() >= opts.iterations;
    });
  }

  if (opts.use_cuda) {
    cudaDeviceSynchronize();
  }

  process_results(results, model_cpu, models_gpu, opts.verbosity);
}
}  // namespace starpu_server
