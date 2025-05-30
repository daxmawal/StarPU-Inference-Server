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
    const torch::Tensor& output_ref, const unsigned int iterations_per_worker,
    const std::map<int, std::vector<int>>&
        device_workers)  // <-- pass by const ref
{
  // Pre-generate a small pool of random input tensors to avoid reallocation
  std::vector<std::vector<torch::Tensor>> pregen_inputs;
  pregen_inputs.reserve(NUM_PREGENERATED_INPUTS);

  for (int i = 0; i < NUM_PREGENERATED_INPUTS; ++i) {
    pregen_inputs.emplace_back(
        generate_random_inputs(opts.input_shapes, opts.input_types));
  }

  // RNG setup to randomly pick pre-generated inputs for warmup jobs
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<> dist(0, NUM_PREGENERATED_INPUTS - 1);

  unsigned int job_id = 0;

  // Iterate over each device ID and its associated StarPU worker IDs
  for (const auto& [device_id, worker_ids] : device_workers) {
    for (int worker_id : worker_ids) {
      for (unsigned int i = 0; i < iterations_per_worker; ++i) {
        auto job = std::make_shared<InferenceJob>();

        const auto& chosen_inputs = pregen_inputs[dist(rng)];
        job->input_tensors = chosen_inputs;

        job->input_types.clear();
        for (const auto& t : chosen_inputs) {
          job->input_types.emplace_back(t.scalar_type());
        }

        job->output_tensor = torch::empty_like(output_ref);
        job->job_id = job_id++;
        job->fixed_worker_id = worker_id;
        job->start_time = std::chrono::high_resolution_clock::now();
        job->timing_info.enqueued_time = job->start_time;

        log_trace(
            opts.verbosity, "[Warmup] Job ID " + std::to_string(job->job_id) +
                                ", Iteration " + std::to_string(i + 1) + "/" +
                                std::to_string(iterations_per_worker) +
                                ", device ID " + std::to_string(device_id) +
                                ", worker ID " + std::to_string(worker_id));

        queue.push(job);
      }
    }
  }

  queue.shutdown();
}

// =============================================================================
// run_warmup_phase: sets up client/server threads to run warmup tasks
// =============================================================================
void
run_warmup_phase(
    const ProgramOptions& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const torch::Tensor& output_ref, const unsigned int iterations_per_worker)
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

  const auto device_workers =
      starpu.get_cuda_workers_by_device(opts.device_ids);

  // Launch client thread to feed warmup jobs
  std::thread client([&]() {
    client_worker_warmup(
        warmup_queue, opts, output_ref, iterations_per_worker, device_workers);
  });

  // Wait for client thread to finish pushing jobs
  client.join();

  // Count the total number of CUDA workers
  int total_worker_count = 0;
  for (const auto& [device_id, worker_list] : device_workers) {
    total_worker_count += static_cast<int>(worker_list.size());
  }

  // Wait until all warmup jobs are marked completed
  {
    std::unique_lock<std::mutex> lock(dummy_mutex);
    const size_t total_jobs =
        static_cast<size_t>(iterations_per_worker) * total_worker_count;
    dummy_cv.wait(
        lock, [&]() { return dummy_completed_jobs.load() >= total_jobs; });
  }

  // Join server thread
  server.join();
}
