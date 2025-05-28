#include "inference_runner.hpp"

#include <random>

#include "Inference_queue.hpp"
#include "exceptions.hpp"
#include "inference_task.hpp"
#include "inference_validator.hpp"
#include "input_generator.hpp"
#include "server_worker.hpp"
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


std::string
current_time_formatted(const std::chrono::high_resolution_clock::time_point& tp)
{
  using namespace std::chrono;

  auto now_ms = time_point_cast<milliseconds>(tp);
  auto value = now_ms.time_since_epoch();

  auto h = duration_cast<std::chrono::hours>(value);
  value -= h;
  auto m = duration_cast<std::chrono::minutes>(value);
  value -= m;
  auto s = duration_cast<std::chrono::seconds>(value);
  value -= s;
  auto ms = duration_cast<std::chrono::milliseconds>(value);

  std::ostringstream oss;
  oss << std::setfill('0') << std::setw(2) << h.count() % 24 << ":"
      << std::setfill('0') << std::setw(2) << m.count() << ":"
      << std::setfill('0') << std::setw(2) << s.count() << "."
      << std::setfill('0') << std::setw(3) << ms.count();

  return oss.str();
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

    log_trace(
        opts.verbosity,
        "[Inference] Job ID " + std::to_string(job->job_id) + ", Iteration " +
            std::to_string(i + 1) + "/" + std::to_string(iterations) +
            ", Enqueued at " +
            current_time_formatted(job->timing_info.enqueued_time));

    queue.push(job);

    std::this_thread::sleep_for(std::chrono::milliseconds(opts.delay_ms));
  }

  queue.shutdown();
}

std::tuple<
    torch::jit::script::Module, std::vector<torch::jit::script::Module>,
    torch::Tensor>
load_model_and_reference_output(const ProgramOptions& opts)
{
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  models_gpu.reserve(opts.device_ids.size());
  torch::Tensor output_ref;

  try {
    model_cpu = torch::jit::load(opts.model_path);

    if (opts.use_cuda) {
      for (auto i = 0; i < opts.device_ids.size(); ++i) {
        torch::jit::script::Module model_gpu = model_cpu.clone();
        const torch::Device device(torch::kCUDA, opts.device_ids[i]);
        model_gpu.to(device);
        models_gpu.emplace_back(std::move(model_gpu));
      }
    }

    auto inputs = generate_random_inputs(opts.input_shapes, opts.input_types);
    std::vector<torch::IValue> input_ivalues(inputs.begin(), inputs.end());

    output_ref = model_cpu.forward(input_ivalues).toTensor();
  }
  catch (const c10::Error& e) {
    log_error(
        std::string("Failed to load model or run reference inference: ") +
        e.what());
    throw;
  }

  return {model_cpu, models_gpu, output_ref};
}

void
run_inference_loop(const ProgramOptions& opts, StarPUSetup& starpu)
{
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  torch::Tensor output_ref;

  try {
    std::tie(model_cpu, models_gpu, output_ref) =
        load_model_and_reference_output(opts);
  }
  catch (...) {
    return;
  }

  const unsigned int warmup_iterations = NUM_WARMUP_ITERATIONS;

  log_info(
      opts.verbosity, "Starting warmup with " +
                          std::to_string(warmup_iterations) +
                          " iterations per CUDA device...");
  run_warmup_phase(
      opts, starpu, model_cpu, models_gpu, output_ref, warmup_iterations);
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
      log_error("[Client] Job " + std::to_string(r.job_id) + " failed.");
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

      std::ostringstream oss;
      oss << std::fixed << std::setprecision(3);
      oss << "Job " << r.job_id << " done. Latency = " << r.latency_ms
          << " ms | " << "Queue = " << queue_duration.count() << " ms, "
          << "Submit = " << submit_duration.count() << " ms, "
          << "Scheduling = " << scheduling_duration.count() << " ms, "
          << "Codelet = " << codelet_duration.count() << " ms, "
          << "Inference = " << inference_duration.count() << " ms, "
          << "Callback = " << callback_duration.count() << " ms";
      log_stats(opts.verbosity, oss.str());

      validate_inference_result(r, model_cpu, opts.verbosity);
    }
  }
}