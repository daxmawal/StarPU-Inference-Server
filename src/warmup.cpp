#include "warmup.hpp"

#include <random>
#include <thread>

#include "Inference_queue.hpp"
#include "exceptions.hpp"
#include "inference_task.hpp"
#include "inference_validator.hpp"
#include "input_generator.hpp"
#include "server_worker.hpp"

constexpr int NUM_PREGENERATED_INPUTS = 2;

// =============================================================================
// client_worker_warmup: generates warmup jobs and pushes them to the queue
// =============================================================================
void
client_worker_warmup(
    InferenceQueue& queue, const ProgramOptions& opts,
    const torch::Tensor& output_ref, const unsigned int iterations_per_device)
{
  // Pre-generate a few random input tensors to reuse
  std::vector<std::vector<torch::Tensor>> pregen_inputs;
  pregen_inputs.reserve(NUM_PREGENERATED_INPUTS);

  for (int i = 0; i < NUM_PREGENERATED_INPUTS; ++i) {
    pregen_inputs.emplace_back(
        generate_random_inputs(opts.input_shapes, opts.input_types));
  }

  // RNG setup to select randomly among pre-generated inputs
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<> dist(0, NUM_PREGENERATED_INPUTS - 1);

  // Generate and enqueue warmup jobs
  unsigned int job_id = 0;
  for (size_t device_index = 0; device_index < opts.device_ids.size();
       ++device_index) {
    unsigned int device_id = opts.device_ids[device_index];

    for (unsigned int i = 0; i < iterations_per_device; ++i) {
      auto job = std::make_shared<InferenceJob>();

      auto index = dist(rng);
      if (index < 0 || static_cast<size_t>(index) >= pregen_inputs.size()) {
        throw std::runtime_error("Generated index is out of range.");
      }
      const auto& chosen_inputs = pregen_inputs[static_cast<size_t>(index)];

      job->input_tensors = chosen_inputs;

      // Infer input types from tensors
      job->input_types.clear();
      for (const auto& t : chosen_inputs) {
        job->input_types.emplace_back(t.scalar_type());
      }

      // Create output tensor with same shape/type as reference
      job->output_tensor = torch::empty_like(output_ref);
      job->job_id = job_id++;
      job->fixed_worker_id = static_cast<int>(device_id);
      job->start_time = std::chrono::high_resolution_clock::now();
      job->timing_info.enqueued_time = job->start_time;

      log_trace(
          opts.verbosity, "[Warmup] Job ID " + std::to_string(job->job_id) +
                              ", Iteration " + std::to_string(i + 1) + "/" +
                              std::to_string(iterations_per_device) +
                              ", device ID " + std::to_string(device_id));

      queue.push(job);
    }
  }

  queue.shutdown();  // Signal the end of job submission
}

// =============================================================================
// run_warmup_phase: sets up client/server threads to run warmup tasks
// =============================================================================
void
run_warmup_phase(
    const ProgramOptions& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const torch::Tensor& output_ref, const unsigned int iterations_per_device)
{
  if (!opts.use_cuda) {
    return;  // No warmup needed without GPU
  }

  // Dummy resources (since warmup results aren't needed)
  InferenceQueue warmup_queue;
  std::atomic<unsigned int> dummy_completed_jobs = 0;
  std::mutex dummy_mutex;
  std::mutex dummy_results_mutex;
  std::condition_variable dummy_cv;
  std::vector<InferenceResult> dummy_results;

  // Launch server thread
  ServerWorker worker(
      warmup_queue, model_cpu, models_gpu, starpu, opts, dummy_results,
      dummy_results_mutex, dummy_completed_jobs, dummy_cv);

  std::thread server(&ServerWorker::run, &worker);

  // Launch client thread to feed warmup jobs
  std::thread client([&]() {
    client_worker_warmup(warmup_queue, opts, output_ref, iterations_per_device);
  });

  // Wait for client thread to finish pushing jobs
  client.join();

  // Wait until all warmup jobs are marked completed
  {
    std::unique_lock<std::mutex> lock(dummy_mutex);
    const size_t total_jobs =
        static_cast<size_t>(iterations_per_device) * opts.device_ids.size();
    dummy_cv.wait(
        lock, [&]() { return dummy_completed_jobs.load() >= total_jobs; });
  }

  // Join server thread
  server.join();
}
