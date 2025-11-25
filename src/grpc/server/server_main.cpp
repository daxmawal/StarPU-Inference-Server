#include <hwloc.h>
#include <starpu.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "core/inference_runner.hpp"
#include "core/starpu_setup.hpp"
#include "inference_service.hpp"
#include "monitoring/metrics.hpp"
#include "starpu_task_worker/inference_queue.hpp"
#include "starpu_task_worker/starpu_task_worker.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/client_utils.hpp"
#include "utils/config_loader.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"
#include "utils/perf_observer.hpp"
#include "utils/runtime_config.hpp"

namespace {
struct ServerContext {
  starpu_server::InferenceQueue* queue_ptr = nullptr;
  std::unique_ptr<grpc::Server> server;
  std::mutex stop_mutex;
  std::condition_variable stop_cv;
  std::atomic<bool> stop_requested{false};
};

auto
server_context() -> ServerContext&
{
  static ServerContext ctx;
  return ctx;
}

auto
shell_quote(const std::string& value) -> std::string
{
  std::string quoted;
  quoted.reserve(value.size() + 2);
  quoted.push_back('\'');
  for (char character : value) {
    if (character == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(character);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

auto
candidate_plot_scripts(const starpu_server::RuntimeConfig& opts)
    -> std::vector<std::filesystem::path>
{
  std::vector<std::filesystem::path> candidates;
  candidates.emplace_back("scripts/plot_batch_summary.py");
  if (!opts.config_path.empty()) {
    const auto config_dir =
        std::filesystem::path(opts.config_path).parent_path();
    candidates.emplace_back(config_dir / "scripts/plot_batch_summary.py");
  }
  std::error_code exe_ec;
  const auto exe_path = std::filesystem::read_symlink("/proc/self/exe", exe_ec);
  if (!exe_ec) {
    candidates.emplace_back(
        std::filesystem::path(exe_path).parent_path() /
        "../scripts/plot_batch_summary.py");
  }
  return candidates;
}

auto
locate_plot_script(const starpu_server::RuntimeConfig& opts)
    -> std::optional<std::filesystem::path>
{
  for (const auto& candidate : candidate_plot_scripts(opts)) {
    if (candidate.empty()) {
      continue;
    }
    auto resolved = candidate;
    if (!resolved.is_absolute()) {
      std::error_code abs_ec;
      const auto absolute = std::filesystem::absolute(resolved, abs_ec);
      if (!abs_ec) {
        resolved = absolute;
      }
    }
    std::error_code exists_ec;
    if (std::filesystem::exists(resolved, exists_ec) && !exists_ec) {
      return resolved;
    }
  }
  return std::nullopt;
}

auto
plots_output_path(const std::filesystem::path& summary_path)
    -> std::filesystem::path
{
  auto filename = summary_path.stem().string();
  if (const auto pos = filename.rfind("_summary"); pos != std::string::npos) {
    filename.erase(pos);
  }
  filename += "_plots.png";
  auto output = summary_path;
  output.replace_filename(filename);
  return output;
}

void
run_trace_plots_if_enabled(const starpu_server::RuntimeConfig& opts)
{
  if (!opts.batching.trace_enabled) {
    return;
  }

  const auto& tracer = starpu_server::BatchingTraceLogger::instance();
  const auto summary_path_opt = tracer.summary_file_path();
  if (!summary_path_opt) {
    starpu_server::log_warning(
        "Tracing was enabled but no batching_trace_summary.csv was produced; "
        "skipping plot generation.");
    return;
  }

  const auto& summary_path = *summary_path_opt;
  if (std::error_code err_code;
      !std::filesystem::exists(summary_path, err_code) || err_code) {
    starpu_server::log_warning(std::format(
        "Tracing summary file '{}' not found; skipping plot generation.",
        summary_path.string()));
    return;
  }

  const auto script_path = locate_plot_script(opts);
  if (!script_path) {
    starpu_server::log_warning(
        "Unable to locate scripts/plot_batch_summary.py; skipping plot "
        "generation.");
    return;
  }

  const auto output_path = plots_output_path(summary_path);
  const std::string command = std::format(
      "python3 {} {} --output {}", shell_quote(script_path->string()),
      shell_quote(summary_path.string()), shell_quote(output_path.string()));
  const int return_code = std::system(command.c_str());
  if (return_code != 0) {
    starpu_server::log_warning(std::format(
        "Failed to generate batching latency plots; command '{}' exited with "
        "code {}.",
        command, return_code));
  } else {
    starpu_server::log_info(
        opts.verbosity,
        std::format(
            "Batching latency plots written to '{}'.", output_path.string()));
  }
}

void
run_startup_throughput_probe(
    const starpu_server::RuntimeConfig& opts,
    starpu_server::StarPUSetup& starpu, torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<torch::Tensor>& reference_outputs)
{
  using namespace std::chrono_literals;

  if (!opts.devices.use_cuda) {
    starpu_server::log_info(
        opts.verbosity,
        "[Throughput] Startup throughput probe skipped: CUDA is disabled.");
    return;
  }

  if (models_gpu.empty()) {
    starpu_server::log_warning(
        "[Throughput] Startup throughput probe skipped: no CUDA model "
        "replicas available.");
    return;
  }

  const auto cuda_workers =
      starpu_server::StarPUSetup::get_worker_ids_by_type(STARPU_CUDA_WORKER);
  if (cuda_workers.empty()) {
    starpu_server::log_warning(
        "[Throughput] Startup throughput probe skipped: no CUDA workers "
        "detected.");
    return;
  }

  const int max_batch_size = std::max(1, opts.batching.max_batch_size);
  const int worker_count = std::max(1, static_cast<int>(cuda_workers.size()));
  constexpr double kMinTargetSeconds = 10.0;
  constexpr double kTargetSeconds = 15.0;
  constexpr int kCalibrationMultiplier = 10;
  constexpr int kFallbackBatchesPerWorker = 150;
  constexpr std::string_view kThroughputSuffix = "_configuration.txt";

  struct ProbeOutcome {
    std::optional<starpu_server::perf_observer::Snapshot> snapshot;
    int completed_jobs = 0;
    bool completed_all = false;
  };

  auto run_probe_once = [&](int request_count,
                            bool show_progress) -> ProbeOutcome {
    ProbeOutcome outcome{};
    if (request_count <= 0) {
      return outcome;
    }

    starpu_server::log_info(
        opts.verbosity,
        std::format(
            "[Throughput] Running {} probe with {} synthetic request(s) "
            "(workers={}, max_batch_size={})...",
            show_progress ? "duration-calibrated" : "calibration",
            request_count, worker_count, max_batch_size));

    starpu_server::perf_observer::reset();

    try {
      starpu_server::InferenceQueue queue;
      std::vector<starpu_server::InferenceResult> throwaway_results;
      std::mutex results_mutex;
      std::atomic<int> completed_jobs{0};
      std::condition_variable all_done_cv;
      std::mutex all_done_mutex;
      std::atomic<bool> stop_progress{false};

      starpu_server::StarPUTaskRunnerConfig config{};
      config.queue = &queue;
      config.model_cpu = &model_cpu;
      config.models_gpu = &models_gpu;
      config.starpu = &starpu;
      config.opts = &opts;
      config.results = &throwaway_results;
      config.results_mutex = &results_mutex;
      config.completed_jobs = &completed_jobs;
      config.all_done_cv = &all_done_cv;

      std::string default_model_name = opts.name;
      if (default_model_name.empty() && !opts.models.empty()) {
        default_model_name = opts.models[0].name;
      }

      starpu_server::StarPUTaskRunner worker(config);
      std::jthread worker_thread(
          &starpu_server::StarPUTaskRunner::run, &worker);

      std::jthread client_thread([&, default_model_name]() {
        std::mt19937 rng;
        if (opts.seed.has_value()) {
          rng.seed(*opts.seed);
          torch::manual_seed(*opts.seed);
        } else {
          rng.seed(std::random_device{}());
        }

        auto pregen_inputs = starpu_server::client_utils::pre_generate_inputs(
            opts, std::max<std::size_t>(1, opts.batching.pregen_inputs));

        for (int request_id = 0; request_id < request_count; ++request_id) {
          const auto& inputs = starpu_server::client_utils::pick_random_input(
              pregen_inputs, rng);
          auto job = starpu_server::client_utils::create_job(
              inputs, reference_outputs, request_id, {}, {},
              default_model_name);
          job->set_gpu_only(true);

          const auto enqueued_now = std::chrono::high_resolution_clock::now();
          job->timing_info().enqueued_time = enqueued_now;
          job->timing_info().last_enqueued_time = enqueued_now;

          if (!queue.push(std::move(job))) {
            starpu_server::log_warning(
                "[Throughput] Failed to enqueue job: queue shutting down");
            break;
          }
        }

        queue.shutdown();
      });

      std::jthread progress_thread;
      if (show_progress) {
        progress_thread = std::jthread([&]() {
          constexpr int bar_width = 20;
          int last_step = -1;
          while (!stop_progress.load(std::memory_order_relaxed)) {
            const int done = completed_jobs.load(std::memory_order_relaxed);
            const int pct = std::clamp(done * 100 / request_count, 0, 100);
            const int step = pct / 5;  // update every 5%
            if (step != last_step) {
              last_step = step;
              const int filled =
                  std::clamp(pct * bar_width / 100, 0, bar_width);
              std::string bar(bar_width, ' ');
              std::fill_n(bar.begin(), filled, '#');
              starpu_server::log_info(
                  opts.verbosity,
                  std::format(
                      "[Throughput] Progress [{}] {:3d}% ({}/{})", bar, pct,
                      done, request_count));
            }
            if (pct >= 100) {
              break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
          }
        });
      }

      const auto deadline = std::chrono::steady_clock::now() + 60s;
      {
        std::unique_lock lock(all_done_mutex);
        outcome.completed_all = all_done_cv.wait_until(lock, deadline, [&]() {
          return completed_jobs.load(std::memory_order_acquire) >=
                 request_count;
        });
      }

      queue.shutdown();
      stop_progress.store(true, std::memory_order_relaxed);
      if (progress_thread.joinable()) {
        progress_thread.join();
      }
      if (client_thread.joinable()) {
        client_thread.join();
      }
      if (worker_thread.joinable()) {
        worker_thread.join();
      }
      starpu_task_wait_for_all();

      outcome.completed_jobs = completed_jobs.load(std::memory_order_acquire);
      outcome.snapshot = starpu_server::perf_observer::snapshot();
      if (outcome.snapshot.has_value()) {
        starpu_server::log_info(
            opts.verbosity,
            std::format(
                "Throughput: {} inference(s) in {:.3f} s -> {:.2f} "
                "infer/s",
                outcome.snapshot->total_inferences,
                outcome.snapshot->duration_seconds,
                outcome.snapshot->throughput));
      } else {
        starpu_server::log_warning(
            "Throughput probe did not yield a usable snapshot.");
      }

      if (!outcome.completed_all) {
        starpu_server::log_warning(std::format(
            "Startup throughput probe timed out; completed {} of {} "
            "request(s).",
            outcome.completed_jobs, request_count));
      }
    }
    catch (const std::exception& e) {
      starpu_server::log_warning(std::format(
          "Throughput probe failed (continuing startup): {}", e.what()));
    }

    starpu_server::perf_observer::reset();
    return outcome;
  };

  const int calibration_requests =
      std::max(1, worker_count * max_batch_size * kCalibrationMultiplier);
  const auto calibration = run_probe_once(calibration_requests, false);

  double estimated_throughput = 0.0;
  if (calibration.snapshot && calibration.snapshot->throughput > 0.0) {
    estimated_throughput = calibration.snapshot->throughput;
  }
  if (estimated_throughput <= 0.0) {
    estimated_throughput = static_cast<double>(
        worker_count * max_batch_size * kFallbackBatchesPerWorker);
    starpu_server::log_warning(
        "[Throughput] Calibration failed; falling back to heuristic "
        "throughput estimate.");
  }

  const double target_seconds = kTargetSeconds;
  double expected_requests = estimated_throughput * target_seconds;
  if (estimated_throughput > 0.0) {
    const double estimated_duration = expected_requests / estimated_throughput;
    if (estimated_duration < kMinTargetSeconds) {
      expected_requests = estimated_throughput * kMinTargetSeconds;
    }
  }

  const auto clamped_requests = std::clamp(
      static_cast<long long>(std::llround(std::ceil(expected_requests))), 1LL,
      static_cast<long long>(std::numeric_limits<int>::max()));
  const auto synthetic_requests = static_cast<int>(clamped_requests);

  starpu_server::log_info(
      opts.verbosity,
      std::format(
          "[Throughput] Targeting ≥{:.1f}s runtime "
          "(estimate {:.1f}s @ {:.2f} infer/s) with {} synthetic request(s).",
          kMinTargetSeconds,
          synthetic_requests / std::max(estimated_throughput, 1e-6),
          estimated_throughput, synthetic_requests));

  const auto final_outcome = run_probe_once(synthetic_requests, true);

  try {
    const auto config_suffix = [&]() -> std::string {
      if (opts.config_path.empty()) {
        return "unnamed";
      }
      const auto path = std::filesystem::path(opts.config_path);
      auto stem = path.stem().string();
      if (stem.empty()) {
        stem = "unnamed";
      }
      return stem;
    }();

    const auto filename =
        std::filesystem::path(config_suffix + std::string(kThroughputSuffix));
    if (final_outcome.snapshot) {
      std::ofstream out(filename, std::ios::trunc);
      if (out) {
        out << std::format("{:.6f}\n", final_outcome.snapshot->throughput);
        starpu_server::log_info(
            opts.verbosity,
            std::format(
                "[Throughput] Wrote measured throughput {:.2f} infer/s to {}",
                final_outcome.snapshot->throughput, filename.string()));
      } else {
        starpu_server::log_warning(std::format(
            "[Throughput] Unable to write throughput file {}",
            filename.string()));
      }
    }
  }
  catch (const std::exception& e) {
    starpu_server::log_warning(std::format(
        "[Throughput] Failed to persist throughput measurement: {}", e.what()));
  }
}
}  // namespace

void
signal_handler(int /*signal*/)
{
  server_context().stop_requested.store(true);
}

auto
handle_program_arguments(std::span<char const* const> args)
    -> starpu_server::RuntimeConfig
{
  const char* config_path = nullptr;

  auto remaining = args.subspan(1);
  auto require_value = [&](std::string_view flag) {
    if (remaining.empty() || remaining.front() == nullptr) {
      starpu_server::log_fatal(
          std::format("Missing value for {} argument.\n", flag));
    }
    const char* value = remaining.front();
    remaining = remaining.subspan(1);
    return value;
  };

  while (!remaining.empty()) {
    const char* raw_arg = remaining.front();
    remaining = remaining.subspan(1);

    if (raw_arg == nullptr) {
      starpu_server::log_fatal("Unexpected null program argument.\n");
    }

    std::string_view arg{raw_arg};
    if (arg == "--config" || arg == "-c") {
      config_path = require_value(arg);
      continue;
    }
    starpu_server::log_fatal(std::format(
        "Unknown argument '{}'. Only --config/-c is supported; all other "
        "settings must live in the YAML file.\n",
        arg));
  }

  if (config_path == nullptr) {
    starpu_server::log_fatal("Missing required --config argument.\n");
  }

  starpu_server::RuntimeConfig cfg = starpu_server::load_config(config_path);
  cfg.config_path = config_path;

  if (!cfg.valid) {
    starpu_server::log_fatal("Invalid configuration file.\n");
  }

  log_info(cfg.verbosity, std::format("__cplusplus = {}", __cplusplus));
  log_info(cfg.verbosity, std::format("LibTorch version: {}", TORCH_VERSION));
  log_info(cfg.verbosity, std::format("Scheduler       : {}", cfg.scheduler));
  if (!cfg.name.empty()) {
    log_info(cfg.verbosity, std::format("Configuration   : {}", cfg.name));
  }

  return cfg;
}

auto
prepare_models_and_warmup(
    const starpu_server::RuntimeConfig& opts,
    starpu_server::StarPUSetup& starpu)
    -> std::tuple<
        torch::jit::script::Module, std::vector<torch::jit::script::Module>,
        std::vector<torch::Tensor>>
{
  auto models = starpu_server::load_model_and_reference_output(opts);
  if (!models) {
    throw starpu_server::ModelLoadingException(
        "Failed to load model or reference outputs");
  }
  auto [model_cpu, models_gpu, reference_outputs] = std::move(*models);
  starpu_server::run_warmup(
      opts, starpu, model_cpu, models_gpu, reference_outputs);
  return {model_cpu, models_gpu, reference_outputs};
}

void
launch_threads(
    const starpu_server::RuntimeConfig& opts,
    starpu_server::StarPUSetup& starpu, torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    std::vector<torch::Tensor>& reference_outputs)
{
  static starpu_server::InferenceQueue queue;
  auto& server_ctx = server_context();
  server_ctx.queue_ptr = &queue;

  std::jthread notifier_thread([&server_ctx]() {
    constexpr auto kNotifierSleep = std::chrono::milliseconds(10);
    while (!server_ctx.stop_requested.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(kNotifierSleep);
    }
    server_ctx.stop_cv.notify_one();
  });

  std::vector<starpu_server::InferenceResult> results;
  std::mutex results_mutex;
  std::atomic completed_jobs{0};
  std::condition_variable all_done_cv;

  starpu_server::StarPUTaskRunnerConfig config{};
  config.queue = &queue;
  config.model_cpu = &model_cpu;
  config.models_gpu = &models_gpu;
  config.starpu = &starpu;
  config.opts = &opts;
  config.results = &results;
  config.results_mutex = &results_mutex;
  config.completed_jobs = &completed_jobs;
  config.all_done_cv = &all_done_cv;
  starpu_server::StarPUTaskRunner worker(config);

  std::jthread worker_thread(&starpu_server::StarPUTaskRunner::run, &worker);
  std::vector<at::ScalarType> expected_input_types;
  if (!opts.models.empty()) {
    expected_input_types.reserve(opts.models[0].inputs.size());
    for (const auto& input : opts.models[0].inputs) {
      expected_input_types.push_back(input.type);
    }
  }
  std::vector<std::vector<int64_t>> expected_input_dims;
  if (!opts.models.empty()) {
    expected_input_dims.reserve(opts.models[0].inputs.size());
    for (const auto& input : opts.models[0].inputs) {
      expected_input_dims.push_back(input.dims);
    }
  }

  std::jthread grpc_thread([&]() {
    std::string default_model_name = opts.name;
    if (default_model_name.empty() && !opts.models.empty()) {
      default_model_name = opts.models[0].name;
    }
    const auto server_options = starpu_server::GrpcServerOptions{
        opts.server_address, opts.batching.max_message_bytes, opts.verbosity,
        std::move(default_model_name)};
    starpu_server::RunGrpcServer(
        queue, reference_outputs, expected_input_types, expected_input_dims,
        opts.batching.max_batch_size, server_options, server_ctx.server);
  });

  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  {
    std::unique_lock lock(server_ctx.stop_mutex);
    server_ctx.stop_cv.wait(
        lock, [] { return server_context().stop_requested.load(); });
  }
  starpu_server::StopServer(server_ctx.server.get());
  if (server_ctx.queue_ptr != nullptr) {
    server_ctx.queue_ptr->shutdown();
  }
  server_ctx.stop_cv.notify_one();
}

auto
worker_type_label(const enum starpu_worker_archtype type) -> std::string
{
  switch (type) {
    case STARPU_CPU_WORKER:
      return "CPU";
    case STARPU_CUDA_WORKER:
      return "CUDA";
    default:
      return std::format("Other({})", static_cast<int>(type));
  }
}

auto
format_cpu_core_ranges(const std::vector<int>& cpus) -> std::string
{
  if (cpus.empty()) {
    return {};
  }

  std::string result;
  auto flush_range = [&](int start, int end) {
    if (!result.empty()) {
      result.push_back(',');
    }
    if (start == end) {
      result += std::to_string(start);
    } else {
      result += std::format("{}-{}", start, end);
    }
  };

  int range_start = cpus.front();
  int previous = range_start;
  for (std::size_t idx = 1; idx < cpus.size(); ++idx) {
    const int core = cpus[idx];
    if (core == previous + 1) {
      previous = core;
    } else {
      flush_range(range_start, previous);
      range_start = previous = core;
    }
  }
  flush_range(range_start, previous);
  return result;
}

auto
describe_cpu_affinity(int worker_id) -> std::string
{
  hwloc_cpuset_t cpuset = starpu_worker_get_hwloc_cpuset(worker_id);
  if (cpuset == nullptr) {
    return {};
  }

  std::vector<int> cores;
  for (int core = hwloc_bitmap_first(cpuset); core != -1;
       core = hwloc_bitmap_next(cpuset, core)) {
    cores.push_back(core);
  }
  hwloc_bitmap_free(cpuset);
  return format_cpu_core_ranges(cores);
}

void
log_worker_inventory(const starpu_server::RuntimeConfig& opts)
{
  const auto total_workers = static_cast<int>(starpu_worker_get_count());
  starpu_server::log_info(
      opts.verbosity,
      std::format("Configured {} StarPU worker(s).", total_workers));

  for (int worker_id = 0; worker_id < total_workers; ++worker_id) {
    const auto type = starpu_worker_get_type(worker_id);
    const int device_id = starpu_worker_get_devid(worker_id);
    const std::string device_label =
        device_id >= 0 ? std::to_string(device_id) : "N/A";
    std::string cpu_affinity;
    if (type == STARPU_CPU_WORKER) {
      const std::string affinity = describe_cpu_affinity(worker_id);
      if (!affinity.empty()) {
        cpu_affinity = std::format(", cores={}", affinity);
      }
    }
    starpu_server::log_info(
        opts.verbosity,
        std::format(
            "Worker {:2d}: type={}, device id={}{}", worker_id,
            worker_type_label(type), device_label, cpu_affinity));
  }
}

auto
main(int argc, char* argv[]) -> int
{
  try {
    starpu_server::RuntimeConfig opts =
        handle_program_arguments({argv, static_cast<size_t>(argc)});
    starpu_server::BatchingTraceLogger::instance().configure_from_runtime(opts);
    const bool metrics_ok = starpu_server::init_metrics(opts.metrics_port);
    if (!metrics_ok) {
      starpu_server::log_warning(
          "Metrics server failed to start; continuing without metrics.");
    }
    starpu_server::StarPUSetup starpu(opts);
    log_worker_inventory(opts);
    auto [model_cpu, models_gpu, reference_outputs] =
        prepare_models_and_warmup(opts, starpu);
    run_startup_throughput_probe(
        opts, starpu, model_cpu, models_gpu, reference_outputs);
    launch_threads(opts, starpu, model_cpu, models_gpu, reference_outputs);
    auto& tracer = starpu_server::BatchingTraceLogger::instance();
    tracer.configure(false, "");
    run_trace_plots_if_enabled(opts);
    starpu_server::shutdown_metrics();
  }
  catch (const starpu_server::InferenceEngineException& e) {
    std::cerr << "\o{33}[1;31m[Inference Error] " << e.what() << "\o{33}[0m\n";
    return 2;
  }
  catch (const std::exception& e) {
    std::cerr << "\o{33}[1;31m[General Error] " << e.what() << "\o{33}[0m\n";
    return -1;
  }

  return 0;
}
