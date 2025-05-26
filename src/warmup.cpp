#include "warmup.hpp"

#include <random>

#include "Inference_queue.hpp"
#include "exceptions.hpp"
#include "inference_task.hpp"
#include "inference_validator.hpp"
#include "input_generator.hpp"

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

      std::cout << "[Warmup] Job ID " << job_id << ", Iteration " << i + 1
                << "/" << iterations_per_device << ", device ID " << device_id
                << std::endl;


      queue.push(job);
    }
  }

  queue.shutdown();
}

void
server_worker_warmup(
    InferenceQueue& queue, torch::jit::script::Module& model_cpu,
    torch::jit::script::Module& model_gpu, StarPUSetup& starpu,
    const ProgramOptions& opts, std::vector<InferenceResult>& results,
    std::mutex& results_mutex, std::atomic<unsigned int>& completed_jobs,
    std::condition_variable& all_done_cv)
{
  while (true) {
    std::shared_ptr<InferenceJob> job;
    queue.wait_and_pop(job);

    if (job->is_shutdown())
      break;

    job->on_complete =
        [id = job->job_id, inputs = job->input_tensors,
         &executed_on = job->executed_on, &timing_info = job->timing_info,
         &device_id = job->device_id, &results, &results_mutex, &completed_jobs,
         &all_done_cv](torch::Tensor result, double latency_ms) {
          {
            std::lock_guard<std::mutex> lock(results_mutex);
            results.push_back(
                {id, inputs, result, latency_ms, executed_on, timing_info,
                 device_id});
          }

          completed_jobs.fetch_add(1);
          all_done_cv.notify_one();
        };

    auto fail_job = [&](const std::string& error_msg) {
      std::cerr << error_msg << std::endl;
      if (job->on_complete) {
        job->on_complete(torch::Tensor(), -1);
      }
    };

    try {
      job->timing_info.dequeued_time =
          std::chrono::high_resolution_clock::now();
      InferenceTask inferenceTask(starpu, job, model_cpu, model_gpu, opts);
      inferenceTask.submit();
    }
    catch (const InferenceEngineException& e) {
      fail_job(
          "[Inference Error] Job " + std::to_string(job->job_id) + ": " +
          e.what());
    }
    catch (const std::exception& e) {
      fail_job(
          "[Unhandled Exception] Job " + std::to_string(job->job_id) + ": " +
          e.what());
    }
  }
}

void
run_warmup_phase(
    const ProgramOptions& opts, StarPUSetup& starpu,
    const torch::jit::script::Module& model_cpu,
    const torch::jit::script::Module& model_gpu,
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

  std::thread server([&]() {
    server_worker_warmup(
        warmup_queue, const_cast<torch::jit::script::Module&>(model_cpu),
        const_cast<torch::jit::script::Module&>(model_gpu), starpu, opts,
        dummy_results, dummy_results_mutex, dummy_completed_jobs, dummy_cv);
  });

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