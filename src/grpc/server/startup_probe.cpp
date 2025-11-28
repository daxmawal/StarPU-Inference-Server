#include "startup_probe.hpp"

#include <starpu.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>

#include "core/starpu_setup.hpp"
#include "starpu_task_worker/inference_queue.hpp"
#include "starpu_task_worker/starpu_task_worker.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/client_utils.hpp"
#include "utils/logger.hpp"
#include "utils/perf_observer.hpp"
#include "utils/runtime_config.hpp"

namespace starpu_server {

namespace {

struct ProbeOutcome {
  std::optional<perf_observer::Snapshot> snapshot;
  int completed_jobs = 0;
  bool completed_all = false;
};

auto
compute_config_signature(const RuntimeConfig& opts) -> std::string
{
  std::ostringstream oss;

  oss << "use_cuda=" << opts.devices.use_cuda << "|";
  oss << "max_batch_size=" << opts.batching.max_batch_size << "|";
  oss << "batch_coalesce_timeout_ms=" << opts.batching.batch_coalesce_timeout_ms
      << "|";
  oss << "pool_size=" << opts.batching.pool_size << "|";
  oss << "dynamic_batching=" << opts.batching.dynamic_batching << "|";
  oss << "device_ids=";
  for (int id : opts.devices.ids) {
    oss << id << ",";
  }
  oss << "|";
  oss << "gpu_models=";
  for (const auto& model : opts.models) {
    oss << model.name << "(";
    for (const auto& input : model.inputs) {
      oss << "[";
      for (int64_t dim : input.dims) {
        oss << dim << ",";
      }
      oss << "]";
    }
    oss << "),";
  }
  oss << "|";
  oss << "starpu_env=";
  for (const auto& [key, value] : opts.starpu_env) {
    oss << key << "=" << value << ",";
  }
  std::hash<std::string> hash_fn;
  auto hash_value = hash_fn(oss.str());
  std::ostringstream hex_stream;
  hex_stream << std::hex << hash_value;
  return hex_stream.str();
}

class ProbeTracePrefixGuard {
 public:
  ProbeTracePrefixGuard(BatchingTraceLogger& tracer, ProbeTraceMode mode)
      : tracer_(tracer), previous_(tracer.probe_mode())
  {
    tracer_.set_probe_mode(mode);
  }
  ProbeTracePrefixGuard(const ProbeTracePrefixGuard&) = delete;
  auto operator=(const ProbeTracePrefixGuard&) -> ProbeTracePrefixGuard& =
                                                      delete;
  ~ProbeTracePrefixGuard() { tracer_.set_probe_mode(previous_); }

 private:
  BatchingTraceLogger& tracer_;
  ProbeTraceMode previous_;
};

}  // namespace

auto
run_startup_throughput_probe_cpu(
    const RuntimeConfig& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<torch::Tensor>& reference_outputs) -> double
{
  using namespace std::chrono_literals;

  double measured_throughput = 0.0;

  const auto cpu_workers =
      StarPUSetup::get_worker_ids_by_type(STARPU_CPU_WORKER);
  if (cpu_workers.empty()) {
    log_warning(
        "[Throughput-CPU] Startup throughput probe skipped: no CPU workers "
        "detected.");
    return measured_throughput;
  }

  const int max_batch_size = std::max(1, opts.batching.max_batch_size);
  const int worker_count = std::max(1, static_cast<int>(cpu_workers.size()));
  constexpr double kMinTargetSeconds = 10.0;
  constexpr double kTargetSeconds = 15.0;
  constexpr int kCalibrationMultiplier = 3;
  constexpr int kFallbackBatchesPerWorker = 20;
  constexpr std::string_view kCpuThroughputSuffix = "_cpu_throughput.txt";
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
  const auto cpu_throughput_filename =
      config_suffix + std::string(kCpuThroughputSuffix);
  const auto cpu_throughput_file = [&]() -> std::filesystem::path {
    if (!opts.batching.file_output_path.empty()) {
      const auto trace_path =
          std::filesystem::path(opts.batching.file_output_path);
      auto output_dir = trace_path.parent_path();
      if (output_dir.empty()) {
        output_dir = ".";
      }
      return output_dir / cpu_throughput_filename;
    }
    return std::filesystem::path(cpu_throughput_filename);
  }();

  double cached_throughput = 0.0;
  {
    std::error_code ec;
    if (std::filesystem::exists(cpu_throughput_file, ec) && !ec) {
      try {
        std::ifstream in(cpu_throughput_file);
        if (in) {
          std::string cached_signature;
          if (std::getline(in, cached_signature)) {
            const auto current_signature = compute_config_signature(opts);
            if (cached_signature == current_signature) {
              in >> cached_throughput;
              if (cached_throughput > 0.0) {
                log_info(
                    opts.verbosity,
                    std::format(
                        "[Throughput-CPU] Cached throughput {:.2f} infer/s "
                        "from "
                        "'{}' "
                        "found (config signature matches), skipping startup "
                        "probe.",
                        cached_throughput, cpu_throughput_file.string()));
                return cached_throughput;
              }
            } else {
              log_info(
                  opts.verbosity,
                  "[Throughput-CPU] Configuration has changed (signature "
                  "mismatch); "
                  "re-running CPU throughput probe.");
            }
          }
        }
      }
      catch (...) {
        log_error("Failed to read cached CPU throughput file");
      }
      log_warning(
          "[Throughput-CPU] Cached throughput file found but unreadable, "
          "signature "
          "mismatch, re-running probe.");
    }
  }

  auto run_probe_once = [&](int request_count,
                            bool show_progress) -> ProbeOutcome {
    ProbeOutcome outcome{};
    if (request_count <= 0) {
      return outcome;
    }

    log_info(
        opts.verbosity,
        std::format(
            "[Throughput-CPU] Running {} probe with {} synthetic request(s) "
            "(workers={}, max_batch_size={})...",
            show_progress ? "duration-calibrated" : "calibration",
            request_count, worker_count, max_batch_size));

    perf_observer::reset();

    try {
      InferenceQueue queue;
      std::vector<InferenceResult> throwaway_results;
      std::mutex results_mutex;
      std::atomic<int> completed_jobs{0};
      std::condition_variable all_done_cv;
      std::mutex all_done_mutex;
      std::atomic<bool> stop_progress{false};

      StarPUTaskRunnerConfig config{};
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

      StarPUTaskRunner worker(config);
      std::jthread worker_thread(&StarPUTaskRunner::run, &worker);

      std::jthread client_thread([&, default_model_name]() {
        std::mt19937 rng;
        if (opts.seed.has_value()) {
          rng.seed(*opts.seed);
          torch::manual_seed(*opts.seed);
        } else {
          rng.seed(std::random_device{}());
        }

        auto pregen_inputs = client_utils::pre_generate_inputs(
            opts, std::max<std::size_t>(1, opts.batching.pregen_inputs));

        for (int request_id = 0; request_id < request_count; ++request_id) {
          const auto& inputs =
              client_utils::pick_random_input(pregen_inputs, rng);
          auto job = client_utils::create_job(
              inputs, reference_outputs, request_id, {}, {},
              default_model_name);
          if (!cpu_workers.empty()) {
            static int cpu_worker_index = 0;
            const int worker_id =
                cpu_workers[cpu_worker_index % cpu_workers.size()];
            job->set_fixed_worker_id(worker_id);
            cpu_worker_index++;
          }

          const auto enqueued_now = std::chrono::high_resolution_clock::now();
          job->timing_info().enqueued_time = enqueued_now;
          job->timing_info().last_enqueued_time = enqueued_now;

          if (!queue.push(std::move(job))) {
            log_warning(
                "[Throughput-CPU] Failed to enqueue job: queue shutting "
                "down");
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
            const int step = pct / 5;
            if (step != last_step) {
              last_step = step;
              const int filled =
                  std::clamp(pct * bar_width / 100, 0, bar_width);
              std::string bar(bar_width, ' ');
              std::fill_n(bar.begin(), filled, '#');
              log_info(
                  opts.verbosity,
                  std::format(
                      "[Throughput-CPU] Progress [{}] {:3d}% ({}/{})", bar, pct,
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
      outcome.snapshot = perf_observer::snapshot();
      if (outcome.snapshot.has_value()) {
        log_info(
            opts.verbosity,
            std::format(
                "[Throughput-CPU] Throughput: {} inference(s) in {:.3f} s "
                "-> {:.2f} infer/s",
                outcome.snapshot->total_inferences,
                outcome.snapshot->duration_seconds,
                outcome.snapshot->throughput));
      } else {
        log_warning("[Throughput-CPU] Probe did not yield a usable snapshot.");
      }

      if (!outcome.completed_all) {
        log_warning(std::format(
            "[Throughput-CPU] Startup throughput probe timed out; completed "
            "{} of {} request(s).",
            outcome.completed_jobs, request_count));
      }
    }
    catch (const std::exception& e) {
      log_warning(std::format(
          "[Throughput-CPU] Probe failed (continuing startup): {}", e.what()));
    }

    perf_observer::reset();
    return outcome;
  };

  const int calibration_requests =
      std::max(1, worker_count * max_batch_size * kCalibrationMultiplier);

  BatchingTraceLogger::instance().enable_probe_measurement();

  ProbeOutcome calibration{};
  {
    ProbeTracePrefixGuard probe_prefix_guard(
        BatchingTraceLogger::instance(), ProbeTraceMode::Calibration);
    calibration = run_probe_once(calibration_requests, false);
  }

  double estimated_throughput = 0.0;
  if (calibration.snapshot && calibration.snapshot->throughput > 0.0) {
    estimated_throughput = calibration.snapshot->throughput;
  }
  if (estimated_throughput <= 0.0) {
    estimated_throughput = static_cast<double>(
        worker_count * max_batch_size * kFallbackBatchesPerWorker);
    log_warning(
        "[Throughput-CPU] Calibration failed; falling back to heuristic "
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

  log_info(
      opts.verbosity,
      std::format(
          "[Throughput-CPU] Targeting ≥{:.1f}s runtime "
          "(estimate {:.1f}s @ {:.2f} infer/s) with {} synthetic request(s).",
          kMinTargetSeconds,
          synthetic_requests / std::max(estimated_throughput, 1e-6),
          estimated_throughput, synthetic_requests));

  measured_throughput = cached_throughput;
  const auto final_outcome =
      measured_throughput > 0.0 ? ProbeOutcome{std::nullopt, 0, true} : [&]() {
        ProbeTracePrefixGuard probe_prefix_guard(
            BatchingTraceLogger::instance(),
            ProbeTraceMode::DurationCalibrated);
        return run_probe_once(synthetic_requests, true);
      }();

  if (!final_outcome.snapshot && measured_throughput <= 0.0) {
    log_warning(
        "[Throughput-CPU] Probe did not produce a throughput snapshot; "
        "nothing to "
        "write.");
    return measured_throughput;
  }

  if (final_outcome.snapshot) {
    measured_throughput = final_outcome.snapshot->throughput;
  }

  try {
    const auto config_signature = compute_config_signature(opts);
    std::ofstream out(cpu_throughput_file, std::ios::trunc);
    if (out) {
      out << config_signature << "\n";
      out << std::format("{:.6f}\n", measured_throughput);
      log_info(
          opts.verbosity,
          std::format(
              "[Throughput-CPU] Wrote measured throughput {:.2f} infer/s to "
              "{} "
              "(config signature: {})",
              measured_throughput, cpu_throughput_file.string(),
              config_signature));
    } else {
      log_warning(std::format(
          "[Throughput-CPU] Unable to write throughput file {}",
          cpu_throughput_file.string()));
    }
  }
  catch (const std::exception& e) {
    log_warning(std::format(
        "[Throughput-CPU] Failed to persist throughput measurement: {}",
        e.what()));
  }

  return measured_throughput;
}

auto
run_startup_throughput_probe(
    const RuntimeConfig& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<torch::Tensor>& reference_outputs) -> double
{
  using namespace std::chrono_literals;

  double measured_throughput = 0.0;

  if (!opts.devices.use_cuda) {
    log_info(
        opts.verbosity,
        "[Throughput] Startup throughput probe skipped: CUDA is disabled.");
    return measured_throughput;
  }

  if (models_gpu.empty()) {
    log_warning(
        "[Throughput] Startup throughput probe skipped: no CUDA model "
        "replicas available.");
    return measured_throughput;
  }

  const auto cuda_workers =
      StarPUSetup::get_worker_ids_by_type(STARPU_CUDA_WORKER);
  if (cuda_workers.empty()) {
    log_warning(
        "[Throughput] Startup throughput probe skipped: no CUDA workers "
        "detected.");
    return measured_throughput;
  }

  const int max_batch_size = std::max(1, opts.batching.max_batch_size);
  const int worker_count = std::max(1, static_cast<int>(cuda_workers.size()));
  constexpr double kMinTargetSeconds = 10.0;
  constexpr double kTargetSeconds = 15.0;
  constexpr int kCalibrationMultiplier = 10;
  constexpr int kFallbackBatchesPerWorker = 150;
  constexpr std::string_view kThroughputSuffix = "_throughput.txt";
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
  const auto throughput_filename =
      config_suffix + std::string(kThroughputSuffix);
  const auto throughput_file = [&]() -> std::filesystem::path {
    if (!opts.batching.file_output_path.empty()) {
      const auto trace_path =
          std::filesystem::path(opts.batching.file_output_path);
      auto output_dir = trace_path.parent_path();
      if (output_dir.empty()) {
        output_dir = ".";
      }
      return output_dir / throughput_filename;
    }
    return std::filesystem::path(throughput_filename);
  }();
  double cached_throughput = 0.0;
  {
    std::error_code ec;
    if (std::filesystem::exists(throughput_file, ec) && !ec) {
      try {
        std::ifstream in(throughput_file);
        if (in) {
          std::string cached_signature;
          if (std::getline(in, cached_signature)) {
            const auto current_signature = compute_config_signature(opts);
            if (cached_signature == current_signature) {
              in >> cached_throughput;
              if (cached_throughput > 0.0) {
                log_info(
                    opts.verbosity,
                    std::format(
                        "[Throughput] Cached throughput {:.2f} infer/s from "
                        "'{}' "
                        "found (config signature matches), skipping startup "
                        "probe.",
                        cached_throughput, throughput_file.string()));
                return cached_throughput;
              }
            } else {
              log_info(
                  opts.verbosity,
                  "[Throughput] Configuration has changed (signature "
                  "mismatch); "
                  "re-running throughput probe.");
            }
          }
        }
      }
      catch (...) {
        log_error("Failed to read cached throughput file");
      }
      log_warning(
          "[Throughput] Cached throughput file found but unreadable, signature "
          "mismatch, re-running probe.");
    }
  }

  auto run_probe_once = [&](int request_count,
                            bool show_progress) -> ProbeOutcome {
    ProbeOutcome outcome{};
    if (request_count <= 0) {
      return outcome;
    }

    log_info(
        opts.verbosity,
        std::format(
            "[Throughput] Running {} probe with {} synthetic request(s) "
            "(workers={}, max_batch_size={})...",
            show_progress ? "duration-calibrated" : "calibration",
            request_count, worker_count, max_batch_size));

    perf_observer::reset();

    try {
      InferenceQueue queue;
      std::vector<InferenceResult> throwaway_results;
      std::mutex results_mutex;
      std::atomic<int> completed_jobs{0};
      std::condition_variable all_done_cv;
      std::mutex all_done_mutex;
      std::atomic<bool> stop_progress{false};

      StarPUTaskRunnerConfig config{};
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

      StarPUTaskRunner worker(config);
      std::jthread worker_thread(&StarPUTaskRunner::run, &worker);

      std::jthread client_thread([&, default_model_name]() {
        std::mt19937 rng;
        if (opts.seed.has_value()) {
          rng.seed(*opts.seed);
          torch::manual_seed(*opts.seed);
        } else {
          rng.seed(std::random_device{}());
        }

        auto pregen_inputs = client_utils::pre_generate_inputs(
            opts, std::max<std::size_t>(1, opts.batching.pregen_inputs));

        for (int request_id = 0; request_id < request_count; ++request_id) {
          const auto& inputs =
              client_utils::pick_random_input(pregen_inputs, rng);
          auto job = client_utils::create_job(
              inputs, reference_outputs, request_id, {}, {},
              default_model_name);
          job->set_gpu_only(true);

          const auto enqueued_now = std::chrono::high_resolution_clock::now();
          job->timing_info().enqueued_time = enqueued_now;
          job->timing_info().last_enqueued_time = enqueued_now;

          if (!queue.push(std::move(job))) {
            log_warning(
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
              log_info(
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
      outcome.snapshot = perf_observer::snapshot();
      if (outcome.snapshot.has_value()) {
        log_info(
            opts.verbosity,
            std::format(
                "Throughput: {} inference(s) in {:.3f} s -> {:.2f} "
                "infer/s",
                outcome.snapshot->total_inferences,
                outcome.snapshot->duration_seconds,
                outcome.snapshot->throughput));
      } else {
        log_warning("Throughput probe did not yield a usable snapshot.");
      }

      if (!outcome.completed_all) {
        log_warning(std::format(
            "Startup throughput probe timed out; completed {} of {} "
            "request(s).",
            outcome.completed_jobs, request_count));
      }
    }
    catch (const std::exception& e) {
      log_warning(std::format(
          "Throughput probe failed (continuing startup): {}", e.what()));
    }

    perf_observer::reset();
    return outcome;
  };

  const int calibration_requests =
      std::max(1, worker_count * max_batch_size * kCalibrationMultiplier);

  BatchingTraceLogger::instance().enable_probe_measurement();

  ProbeOutcome calibration{};
  {
    ProbeTracePrefixGuard probe_prefix_guard(
        BatchingTraceLogger::instance(), ProbeTraceMode::Calibration);
    calibration = run_probe_once(calibration_requests, false);
  }

  double estimated_throughput = 0.0;
  if (calibration.snapshot && calibration.snapshot->throughput > 0.0) {
    estimated_throughput = calibration.snapshot->throughput;
  }
  if (estimated_throughput <= 0.0) {
    estimated_throughput = static_cast<double>(
        worker_count * max_batch_size * kFallbackBatchesPerWorker);
    log_warning(
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

  log_info(
      opts.verbosity,
      std::format(
          "[Throughput] Targeting ≥{:.1f}s runtime "
          "(estimate {:.1f}s @ {:.2f} infer/s) with {} synthetic request(s).",
          kMinTargetSeconds,
          synthetic_requests / std::max(estimated_throughput, 1e-6),
          estimated_throughput, synthetic_requests));

  measured_throughput = cached_throughput;
  const auto final_outcome =
      measured_throughput > 0.0 ? ProbeOutcome{std::nullopt, 0, true} : [&]() {
        ProbeTracePrefixGuard probe_prefix_guard(
            BatchingTraceLogger::instance(),
            ProbeTraceMode::DurationCalibrated);
        return run_probe_once(synthetic_requests, true);
      }();

  if (!final_outcome.snapshot && measured_throughput <= 0.0) {
    log_warning(
        "[Throughput] Probe did not produce a throughput snapshot; nothing to "
        "write.");
    return measured_throughput;
  }

  if (final_outcome.snapshot) {
    measured_throughput = final_outcome.snapshot->throughput;
  }

  try {
    const auto config_signature = compute_config_signature(opts);
    std::ofstream out(throughput_file, std::ios::trunc);
    if (out) {
      out << config_signature << "\n";
      out << std::format("{:.6f}\n", measured_throughput);
      log_info(
          opts.verbosity,
          std::format(
              "[Throughput] Wrote measured throughput {:.2f} infer/s to {} "
              "(config signature: {})",
              measured_throughput, throughput_file.string(), config_signature));
    } else {
      log_warning(std::format(
          "[Throughput] Unable to write throughput file {}",
          throughput_file.string()));
    }
  }
  catch (const std::exception& e) {
    log_warning(std::format(
        "[Throughput] Failed to persist throughput measurement: {}", e.what()));
  }

  return measured_throughput;
}

}  // namespace starpu_server
