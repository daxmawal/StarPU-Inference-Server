#include "warmup.hpp"

#include <random>

#include "Inference_queue.hpp"
#include "exceptions.hpp"
#include "inference_task.hpp"
#include "inference_validator.hpp"
#include "input_generator.hpp"
#include "server_worker.hpp"

constexpr int NUM_PREGENERATED_INPUTS = 2;

void
client_worker_warmup(
    InferenceQueue& queue, const ProgramOptions& opts,
    const torch::Tensor& output_ref, const unsigned int iterations_per_device)
{
  std::vector<std::vector<torch::Tensor>> pregen_inputs;
  pregen_inputs.reserve(NUM_PREGENERATED_INPUTS);

  for (unsigned int i = 0; i < NUM_PREGENERATED_INPUTS; ++i) {
    pregen_inputs.push_back(
        generate_random_inputs(opts.input_shapes, opts.input_types));
  }

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<> dist(0, NUM_PREGENERATED_INPUTS - 1);

  unsigned int job_id = 0;
  for (size_t device_index = 0; device_index < opts.device_ids.size();
       ++device_index) {
    unsigned int device_id = opts.device_ids[device_index];

    for (unsigned int i = 0; i < iterations_per_device; ++i) {
      auto job = std::make_shared<InferenceJob>();

      auto idx = dist(rng);
      TORCH_CHECK(
          idx >= 0 && static_cast<size_t>(idx) < pregen_inputs.size(),
          "Invalid index from RNG");

      const auto& chosen_inputs = pregen_inputs[static_cast<size_t>(idx)];
      job->input_tensors = chosen_inputs;
      job->input_types.clear();
      for (const auto& t : chosen_inputs) {
        job->input_types.emplace_back(t.scalar_type());
      }

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

  queue.shutdown();
}

void
run_warmup_phase(
    const ProgramOptions& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const torch::Tensor& output_ref, const unsigned int iterations_per_device)
{
  if (!opts.use_cuda) {
    return;
  }

  InferenceQueue warmup_queue;
  std::atomic<unsigned int> dummy_completed_jobs = 0;
  std::mutex dummy_mutex;
  std::condition_variable dummy_cv;
  std::vector<InferenceResult> dummy_results;
  std::mutex dummy_results_mutex;

  ServerWorker worker(
      warmup_queue, model_cpu, models_gpu, starpu, opts, dummy_results,
      dummy_results_mutex, dummy_completed_jobs, dummy_cv);

  std::thread server(&ServerWorker::run, &worker);

  std::thread client([&]() {
    client_worker_warmup(warmup_queue, opts, output_ref, iterations_per_device);
  });

  client.join();

  {
    std::unique_lock<std::mutex> lock(dummy_mutex);
    const size_t total_jobs =
        static_cast<size_t>(iterations_per_device) * opts.device_ids.size();
    dummy_cv.wait(
        lock, [&]() { return dummy_completed_jobs.load() >= total_jobs; });
  }

  server.join();
}