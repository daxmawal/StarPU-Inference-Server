#include "inference_runner.hpp"

#include <random>

#include "Inference_queue.hpp"
#include "exceptions.hpp"
#include "inference_task.hpp"
#include "inference_validator.hpp"
#include "input_generator.hpp"
#include "warmup.hpp"

constexpr int NUM_PREGENERATED_INPUTS = 10;
constexpr int NUM_WARMUP_ITERATIONS = 2;

InferenceJob::InferenceJob(
    std::vector<torch::Tensor> inputs, std::vector<at::ScalarType> types,
    unsigned int id, std::function<void(torch::Tensor, int64_t)> callback)
    : input_tensors(std::move(inputs)), input_types(std::move(types)),
      job_id(id), on_complete(std::move(callback)),
      start_time(std::chrono::high_resolution_clock::now())
{
}

std::shared_ptr<InferenceJob>
InferenceJob::make_shutdown_job()
{
  auto job = std::make_shared<InferenceJob>();
  job->is_shutdown_signal_ = true;
  return job;
}

bool
InferenceJob::is_shutdown() const
{
  return is_shutdown_signal_;
}

void
client_worker(
    InferenceQueue& queue, const ProgramOptions& opts,
    const torch::Tensor& output_ref, unsigned int iterations)
{
  std::vector<std::vector<torch::Tensor>> pregen_inputs;
  pregen_inputs.reserve(NUM_PREGENERATED_INPUTS);

  for (int i = 0; i < NUM_PREGENERATED_INPUTS; ++i) {
    pregen_inputs.push_back(
        generate_random_inputs(opts.input_shapes, opts.input_types));
  }

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<> dist(0, NUM_PREGENERATED_INPUTS - 1);

  for (unsigned int i = 0; i < iterations; ++i) {
    auto job = std::make_shared<InferenceJob>();
    auto idx = dist(rng);
    TORCH_CHECK(
        idx >= 0 && static_cast<size_t>(idx) < pregen_inputs.size(),
        "Invalid index from RNG");
    const auto& chosen_inputs = pregen_inputs[static_cast<size_t>(idx)];
    job->input_tensors = chosen_inputs;

    job->input_types.clear();
    for (const auto& t : job->input_tensors) {
      job->input_types.emplace_back(t.scalar_type());
    }

    job->output_tensor = torch::empty_like(output_ref);
    job->job_id = i;
    job->start_time = std::chrono::high_resolution_clock::now();
    job->timing_info.enqueued_time = job->start_time;
    queue.push(job);

    std::this_thread::sleep_for(std::chrono::milliseconds(opts.delay_ms));
  }

  queue.shutdown();
}

void
server_worker(
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

std::tuple<
    torch::jit::script::Module, torch::jit::script::Module, torch::Tensor>
load_model_and_reference_output(const ProgramOptions& opts)
{
  torch::jit::script::Module model_cpu;
  torch::jit::script::Module model_gpu;
  torch::Tensor output_ref;

  try {
    model_cpu = torch::jit::load(opts.model_path);

    if (opts.use_cuda) {
      model_gpu = model_cpu.clone();
      const torch::Device device(torch::kCUDA, 0);
      model_gpu.to(device);
    }

    auto inputs = generate_random_inputs(opts.input_shapes, opts.input_types);
    std::vector<torch::IValue> input_ivalues(inputs.begin(), inputs.end());

    output_ref = model_cpu.forward(input_ivalues).toTensor();
  }
  catch (const c10::Error& e) {
    std::cerr << "[Error] Failed to load model or run reference inference: "
              << e.what() << std::endl;
    throw;
  }

  return {model_cpu, model_gpu, output_ref};
}

void
run_inference_loop(const ProgramOptions& opts, StarPUSetup& starpu)
{
  torch::jit::script::Module model_cpu;
  torch::jit::script::Module model_gpu;
  torch::Tensor output_ref;

  try {
    std::tie(model_cpu, model_gpu, output_ref) =
        load_model_and_reference_output(opts);
  }
  catch (...) {
    return;
  }

  const unsigned int warmup_iterations = NUM_WARMUP_ITERATIONS;
  std::cout << "[Info] Starting warmup with " << warmup_iterations
            << " iterations per CUDA devices...\n";
  run_warmup_phase(
      opts, starpu, model_cpu, model_gpu, output_ref, warmup_iterations);
  std::cout << "[Info] Warmup complete. Proceeding to real inference.\n";

  InferenceQueue queue;
  std::vector<InferenceResult> results;
  std::mutex results_mutex;

  if (opts.iterations > 0) {
    results.reserve(static_cast<size_t>(opts.iterations));
  }
  std::atomic<unsigned int> completed_jobs = 0;
  std::condition_variable all_done_cv;
  std::mutex all_done_mutex;

  std::thread server([&]() {
    server_worker(
        queue, model_cpu, model_gpu, starpu, opts, results, results_mutex,
        completed_jobs, all_done_cv);
  });

  std::thread client(
      [&]() { client_worker(queue, opts, output_ref, opts.iterations); });

  client.join();

  {
    std::unique_lock<std::mutex> lock(all_done_mutex);
    all_done_cv.wait(
        lock, [&]() { return completed_jobs.load() >= opts.iterations; });
  }

  server.join();

  for (const auto& r : results) {
    if (!r.result.defined()) {
      std::cerr << "[Client] Job " << r.job_id << " failed.\n";
    } else {
      auto& t = r.timing_info;
      using duration_f = std::chrono::duration<double, std::milli>;
      auto queue_duration = duration_f(t.dequeued_time - t.enqueued_time);
      auto submit_duration =
          duration_f(t.before_starpu_submitted_time - t.dequeued_time);
      auto scheduling_duration =
          duration_f(t.codelet_start_time - t.before_starpu_submitted_time);
      auto codelet_duration =
          duration_f(t.codelet_end_time - t.codelet_start_time);
      auto inference_duration =
          duration_f(t.callback_start_time - t.inference_start_time);
      auto callback_duration =
          duration_f(t.callback_end_time - t.callback_start_time);

      std::cout << std::fixed << std::setprecision(3);

      std::cout << "[Client] Job " << r.job_id
                << " done. Latency = " << r.latency_ms << " ms | "
                << "Queue = " << queue_duration.count() << " ms, "
                << "Submit = " << submit_duration.count() << " ms, "
                << "Scheduling = " << scheduling_duration.count() << " ms, "
                << "Codelet = " << codelet_duration.count() << " ms, "
                << "Inference = " << inference_duration.count() << " ms, "
                << "Callback = " << callback_duration.count() << " ms\n";

      validate_inference_result(r, model_cpu);
    }
  }
}