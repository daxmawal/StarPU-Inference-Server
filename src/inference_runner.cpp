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

#include "Inference_queue.hpp"
#include "args_parser.hpp"
#include "inference_validator.hpp"
#include "input_generator.hpp"
#include "logger.hpp"
#include "server_worker.hpp"
#include "starpu_setup.hpp"
#include "warmup.hpp"

constexpr int NUM_PREGENERATED_INPUTS = 10;  // Number of reusable input sets
constexpr int NUM_WARMUP_ITERATIONS = 2;     // Warmup runs per GPU
constexpr int HOURS_PER_DAY = 24;

// =============================================================================
// InferenceJob implementation
// =============================================================================
InferenceJob::InferenceJob(
    std::vector<torch::Tensor> inputs, std::vector<at::ScalarType> types,
    unsigned int job_identifier,
    std::function<void(std::vector<torch::Tensor>, double)> callback)
    : input_tensors(std::move(inputs)), input_types(std::move(types)),
      job_id(job_identifier), on_complete(std::move(callback)),
      start_time(std::chrono::high_resolution_clock::now())
{
}

auto
InferenceJob::make_shutdown_job() -> std::shared_ptr<InferenceJob>
{
  auto job = std::make_shared<InferenceJob>();
  job->is_shutdown_signal_ = true;
  return job;
}

auto
InferenceJob::is_shutdown() const -> bool
{
  return is_shutdown_signal_;
}


// =============================================================================
// Utility: Time formatting
// =============================================================================
auto
current_time_formatted(const std::chrono::high_resolution_clock::time_point&
                           time_point) -> std::string
{
  using std::chrono::duration_cast;
  using std::chrono::hours;
  using std::chrono::milliseconds;
  using std::chrono::minutes;
  using std::chrono::seconds;
  using std::chrono::time_point_cast;

  const auto now_ms = time_point_cast<milliseconds>(time_point);
  auto value = now_ms.time_since_epoch();

  const auto hour = duration_cast<hours>(value);
  value -= hour;
  const auto minute = duration_cast<minutes>(value);
  value -= minute;
  const auto second = duration_cast<seconds>(value);
  value -= second;
  const auto milli_second = duration_cast<milliseconds>(value);

  std::ostringstream oss;
  oss << std::setfill('0') << std::setw(2) << hour.count() % HOURS_PER_DAY
      << ":" << std::setfill('0') << std::setw(2) << minute.count() << ":"
      << std::setfill('0') << std::setw(2) << second.count() << "."
      << std::setfill('0') << std::setw(3) << milli_second.count();

  return oss.str();
}


// =============================================================================
// Client thread: generates jobs and feeds the queue
// =============================================================================
void
client_worker(
    InferenceQueue& queue, const ProgramOptions& opts,
    const std::vector<torch::Tensor>& outputs_ref,
    const unsigned int iterations)
{
  std::vector<std::vector<torch::Tensor>> pregen_inputs;
  pregen_inputs.reserve(NUM_PREGENERATED_INPUTS);

  pregen_inputs.reserve(NUM_PREGENERATED_INPUTS);
  std::generate_n(
      std::back_inserter(pregen_inputs), NUM_PREGENERATED_INPUTS, [&]() {
        return generate_random_inputs(opts.input_shapes, opts.input_types);
      });

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<> dist(0, NUM_PREGENERATED_INPUTS - 1);

  for (unsigned int job_id = 0; job_id < iterations; ++job_id) {
    auto job = std::make_shared<InferenceJob>();
    const auto idx = dist(rng);
    TORCH_CHECK(
        idx >= 0 && static_cast<size_t>(idx) < pregen_inputs.size(),
        "Invalid index from RNG");

    const auto& chosen_inputs = pregen_inputs[static_cast<size_t>(idx)];
    job->input_tensors = chosen_inputs;

    job->input_types.reserve(job->input_tensors.size());
    for (const auto& tensor : job->input_tensors) {
      job->input_types.emplace_back(tensor.scalar_type());
    }

    job->outputs_tensors.reserve(outputs_ref.size());
    for (const auto& ref : outputs_ref) {
      job->outputs_tensors.emplace_back(torch::empty_like(ref));
    }

    job->job_id = job_id;
    job->start_time = std::chrono::high_resolution_clock::now();
    job->timing_info.enqueued_time = job->start_time;

    log_trace(
        opts.verbosity,
        "[Inference] Job ID " + std::to_string(job->job_id) + ", Iteration " +
            std::to_string(job_id + 1) + "/" + std::to_string(iterations) +
            ", Enqueued at " +
            current_time_formatted(job->timing_info.enqueued_time));

    queue.push(job);

    std::this_thread::sleep_for(std::chrono::milliseconds(opts.delay_ms));
  }

  queue.shutdown();
}


// =============================================================================
// Load model and generate reference output
// =============================================================================
auto
load_model_and_reference_output(const ProgramOptions& opts)
    -> std::tuple<
        torch::jit::script::Module, std::vector<torch::jit::script::Module>,
        std::vector<torch::Tensor>>
{
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  models_gpu.reserve(opts.device_ids.size());
  std::vector<torch::Tensor> output_refs;

  try {
    model_cpu = torch::jit::load(opts.model_path);

    if (opts.use_cuda) {
      for (const unsigned int device_id : opts.device_ids) {
        torch::jit::script::Module model_gpu = model_cpu.clone();
        const torch::Device device(
            torch::kCUDA, static_cast<c10::DeviceIndex>(device_id));
        model_gpu.to(device);
        models_gpu.emplace_back(std::move(model_gpu));
      }
    }

    const auto inputs =
        generate_random_inputs(opts.input_shapes, opts.input_types);
    const std::vector<torch::IValue> input_ivalues(
        inputs.begin(), inputs.end());
    const auto output = model_cpu.forward(input_ivalues);

    // Manage different types of output
    if (output.isTensor()) {
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
      throw std::runtime_error("Unsupported model output type");
    }
  }
  catch (const c10::Error& e) {
    log_error(
        "Failed to load model or run reference inference: " +
        std::string(e.what()));
    throw;
  }

  return {model_cpu, models_gpu, output_refs};
}


// =============================================================================
// Inference runner: orchestrates the whole process
// =============================================================================
void
run_inference_loop(const ProgramOptions& opts, StarPUSetup& starpu)
{
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> outputs_ref;

  try {
    std::tie(model_cpu, models_gpu, outputs_ref) =
        load_model_and_reference_output(opts);
  }
  catch (...) {
    return;
  }

  log_info(
      opts.verbosity, "Starting warmup with " +
                          std::to_string(NUM_WARMUP_ITERATIONS) +
                          " iterations per CUDA device...");
  run_warmup_phase(
      opts, starpu, model_cpu, models_gpu, outputs_ref, NUM_WARMUP_ITERATIONS);
  log_info(opts.verbosity, "Warmup complete. Proceeding to real inference.\n");

  InferenceQueue queue;
  std::vector<InferenceResult> results;
  std::mutex results_mutex;

  if (opts.iterations > 0) {
    results.reserve(static_cast<size_t>(opts.iterations));
  }

  std::atomic<unsigned int> completed_jobs = 0;
  std::condition_variable all_done_cv;
  std::mutex all_done_mutex;

  ServerWorker worker(
      queue, model_cpu, models_gpu, starpu, opts, results, results_mutex,
      completed_jobs, all_done_cv);
  std::thread server(&ServerWorker::run, &worker);
  std::thread client(
      [&]() { client_worker(queue, opts, outputs_ref, opts.iterations); });

  client.join();

  {
    std::unique_lock<std::mutex> lock(all_done_mutex);
    all_done_cv.wait(
        lock, [&]() { return completed_jobs.load() >= opts.iterations; });
  }

  server.join();

  for (const auto& result : results) {
    if (!result.results[0].defined()) {
      log_error("[Client] Job " + std::to_string(result.job_id) + " failed.");
      continue;
    }

    const auto& time_info = result.timing_info;
    using duration_f = std::chrono::duration<double, std::milli>;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "Job " << result.job_id << " done. Latency = " << result.latency_ms
        << " ms | " << "Queue = "
        << duration_f(time_info.dequeued_time - time_info.enqueued_time).count()
        << " ms, " << "Submit = "
        << duration_f(
               time_info.before_starpu_submitted_time - time_info.dequeued_time)
               .count()
        << " ms, " << "Scheduling = "
        << duration_f(
               time_info.codelet_start_time -
               time_info.before_starpu_submitted_time)
               .count()
        << " ms, " << "Codelet = "
        << duration_f(time_info.codelet_end_time - time_info.codelet_start_time)
               .count()
        << " ms, " << "Inference = "
        << duration_f(
               time_info.callback_start_time - time_info.inference_start_time)
               .count()
        << " ms, " << "Callback = "
        << duration_f(
               time_info.callback_end_time - time_info.callback_start_time)
               .count()
        << " ms";

    log_stats(opts.verbosity, oss.str());
    validate_inference_result(result, model_cpu, opts.verbosity);
  }
}
