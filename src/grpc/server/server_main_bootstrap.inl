void
signal_handler(int /*signal*/)
{
  const int saved_errno = errno;
  signal_stop_requested_flag() = 1;
  const auto notify_fd = signal_stop_notify_fd();
  if (notify_fd >= 0) {
    const std::uint8_t byte = 1;
    const ssize_t write_result =
        ::write(static_cast<int>(notify_fd), &byte, sizeof(byte));
    (void)write_result;
  }
  errno = saved_errno;
}

constexpr auto kShutdownDrainTimeout = std::chrono::seconds(30);
constexpr auto kShutdownDrainWaitStep = std::chrono::milliseconds(250);

enum class ShutdownDrainStageForTest : std::uint8_t {
  Entered,
  CompletedReachedTotal,
  DeadlineReached,
  BeforeWait,
};

struct ShutdownDrainProgressForTest {
  std::size_t total_jobs = 0;
  std::size_t completed_jobs = 0;
};

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START

using ShutdownDrainTimeoutOverrideForTestFn =
    std::chrono::steady_clock::duration (*)();
using ShutdownDrainWaitStepOverrideForTestFn =
    std::chrono::steady_clock::duration (*)();
using ShutdownDrainObserverForTestFn = void (*)(
    ShutdownDrainStageForTest, ShutdownDrainProgressForTest,
    std::chrono::steady_clock::duration);

auto
shutdown_drain_timeout_override_for_test() noexcept
    -> ShutdownDrainTimeoutOverrideForTestFn&
{
  static ShutdownDrainTimeoutOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
shutdown_drain_wait_step_override_for_test() noexcept
    -> ShutdownDrainWaitStepOverrideForTestFn&
{
  static ShutdownDrainWaitStepOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
shutdown_drain_observer_for_test() noexcept -> ShutdownDrainObserverForTestFn&
{
  static ShutdownDrainObserverForTestFn observer_fn = nullptr;
  return observer_fn;
}
#endif  // SONAR_IGNORE_STOP

auto
resolve_shutdown_drain_timeout() -> std::chrono::steady_clock::duration
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = shutdown_drain_timeout_override_for_test();
      override_fn != nullptr) {
    return override_fn();
  }
#endif  // SONAR_IGNORE_STOP
  return kShutdownDrainTimeout;
}

auto
resolve_shutdown_drain_wait_step() -> std::chrono::steady_clock::duration
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = shutdown_drain_wait_step_override_for_test();
      override_fn != nullptr) {
    return override_fn();
  }
#endif  // SONAR_IGNORE_STOP
  return kShutdownDrainWaitStep;
}

void
notify_shutdown_drain_stage_for_test(
    ShutdownDrainStageForTest stage, ShutdownDrainProgressForTest progress,
    std::chrono::steady_clock::duration wait_budget =
        std::chrono::steady_clock::duration::zero())
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto observer_fn = shutdown_drain_observer_for_test();
      observer_fn != nullptr) {
    observer_fn(stage, progress, wait_budget);
  }
#else
  (void)stage;
  (void)progress;
  (void)wait_budget;
#endif  // SONAR_IGNORE_STOP
}

auto
resolve_starpu_scheduler(const starpu_server::RuntimeConfig& opts)
    -> std::string
{
  if (const auto scheduler_env_it =
          opts.starpu_env.find(starpu_server::kStarpuSchedulerEnvVar);
      scheduler_env_it != opts.starpu_env.end()) {
    return std::format("{} (from starpu_env)", scheduler_env_it->second);
  }

  if (const char* env_value =
          std::getenv(starpu_server::kStarpuSchedulerEnvVar.data())) {
    return std::format("{} (from environment)", env_value);
  }

  return std::format("{} (default)", starpu_server::kDefaultStarpuScheduler);
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using HandleProgramArgumentsFatalOverrideForTestFn = void (*)(std::string_view);

auto
handle_program_arguments_fatal_override_for_test() noexcept
    -> HandleProgramArgumentsFatalOverrideForTestFn&
{
  static HandleProgramArgumentsFatalOverrideForTestFn override_fn = nullptr;
  return override_fn;
}
#endif  // SONAR_IGNORE_STOP

[[noreturn]] void
handle_program_arguments_fatal(std::string_view message)
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn =
          handle_program_arguments_fatal_override_for_test();
      override_fn != nullptr) {
    override_fn(message);
    std::terminate();
  }
#endif  // SONAR_IGNORE_STOP
  starpu_server::log_fatal(std::string(message));
}

auto
handle_program_arguments(std::span<char const* const> args)
    -> starpu_server::RuntimeConfig
{
  const char* config_path = nullptr;

  auto remaining = args.subspan(1);
  auto require_value = [&](std::string_view flag) {
    if (remaining.empty() || remaining.front() == nullptr) {
      handle_program_arguments_fatal(
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
      handle_program_arguments_fatal("Unexpected null program argument.\n");
    }

    std::string_view arg{raw_arg};
    if (arg == "--config" || arg == "-c") {
      config_path = require_value(arg);
      continue;
    }
    handle_program_arguments_fatal(std::format(
        "Unknown argument '{}'. Only --config/-c is supported; all other "
        "settings must live in the YAML file.\n",
        arg));
  }

  if (config_path == nullptr) {
    handle_program_arguments_fatal("Missing required --config argument.\n");
  }

  starpu_server::RuntimeConfig cfg = starpu_server::load_config(config_path);

  if (!cfg.valid) {
    handle_program_arguments_fatal("Invalid configuration file.\n");
  }

  log_info(cfg.verbosity, std::format("__cplusplus = {}", __cplusplus));
  log_info(cfg.verbosity, std::format("LibTorch version: {}", TORCH_VERSION));
  log_info(
      cfg.verbosity,
      std::format("StarPU scheduler: {}", resolve_starpu_scheduler(cfg)));
  if (!cfg.name.empty()) {
    log_info(cfg.verbosity, std::format("Configuration   : {}", cfg.name));
  }

  return cfg;
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto
load_model_and_reference_output_override_for_test() noexcept
    -> std::optional<std::tuple<
        torch::jit::script::Module, std::vector<torch::jit::script::Module>,
        std::vector<torch::Tensor>>> (*&)(const starpu_server::RuntimeConfig&)
{
  static std::optional<std::tuple<
      torch::jit::script::Module, std::vector<torch::jit::script::Module>,
      std::vector<torch::Tensor>>> (*override_fn)(
      const starpu_server::RuntimeConfig&) = nullptr;
  return override_fn;
}

auto
run_warmup_override_for_test() noexcept
    -> void (*&)(
        const starpu_server::RuntimeConfig&, starpu_server::StarPUSetup&,
        torch::jit::script::Module&, std::vector<torch::jit::script::Module>&,
        const std::vector<torch::Tensor>&)
{
  static void (*override_fn)(
      const starpu_server::RuntimeConfig&, starpu_server::StarPUSetup&,
      torch::jit::script::Module&, std::vector<torch::jit::script::Module>&,
      const std::vector<torch::Tensor>&) = nullptr;
  return override_fn;
}
#endif  // SONAR_IGNORE_STOP

auto
prepare_models_and_warmup(
    const starpu_server::RuntimeConfig& opts,
    starpu_server::StarPUSetup& starpu)
    -> std::tuple<
        torch::jit::script::Module, std::vector<torch::jit::script::Module>,
        std::vector<torch::Tensor>>
{
  auto models = [&]() {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (const auto override_fn =
            load_model_and_reference_output_override_for_test();
        override_fn != nullptr) {
      return override_fn(opts);
    }
#endif  // SONAR_IGNORE_STOP
    return starpu_server::load_model_and_reference_output(opts);
  }();
  if (!models) {
    throw starpu_server::ModelLoadingException(
        "Failed to load model or reference outputs");
  }
  auto [model_cpu, models_gpu, reference_outputs] = std::move(*models);
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = run_warmup_override_for_test();
      override_fn != nullptr) {
    override_fn(opts, starpu, model_cpu, models_gpu, reference_outputs);
  } else {
    starpu_server::run_warmup(
        opts, starpu, model_cpu, models_gpu, reference_outputs);
  }
#else
  starpu_server::run_warmup(
      opts, starpu, model_cpu, models_gpu, reference_outputs);
#endif  // SONAR_IGNORE_STOP
  return {model_cpu, models_gpu, reference_outputs};
}

auto
make_congestion_config(const starpu_server::RuntimeConfig& cfg)
    -> starpu_server::congestion::Config
{
  using namespace std::chrono;

  starpu_server::congestion::Config out{};
  out.enabled = cfg.congestion.enabled;
  out.latency_slo_ms = cfg.congestion.latency_slo_ms;
  out.queue_latency_budget_ms = cfg.congestion.queue_latency_budget_ms;
  out.queue_latency_budget_ratio = cfg.congestion.queue_latency_budget_ratio;
  out.e2e_warn_ratio = cfg.congestion.e2e_warn_ratio;
  out.e2e_ok_ratio = cfg.congestion.e2e_ok_ratio;
  out.fill_high = cfg.congestion.fill_high;
  out.fill_low = cfg.congestion.fill_low;
  out.rho_high = cfg.congestion.rho_high;
  out.rho_low = cfg.congestion.rho_low;
  out.alpha = cfg.congestion.alpha;
  out.entry_horizon =
      milliseconds(std::max(1, cfg.congestion.entry_horizon_ms));
  out.exit_horizon = milliseconds(std::max(1, cfg.congestion.exit_horizon_ms));
  out.tick_interval =
      milliseconds(std::max(1, cfg.congestion.tick_interval_ms));
  return out;
}

auto
resolve_default_model_name(const starpu_server::RuntimeConfig& opts)
    -> std::string
{
  if (opts.model.has_value() && !opts.model->name.empty()) {
    return opts.model->name;
  }
  return opts.name;
}

auto
make_grpc_server_options(const starpu_server::RuntimeConfig& opts)
    -> starpu_server::GrpcServerOptions
{
  return {opts.server_address, opts.batching.max_message_bytes,
          opts.verbosity,      resolve_default_model_name(opts),
          opts.name,           ""};
}

auto
make_grpc_model_spec(
    const starpu_server::RuntimeConfig& opts,
    std::span<const at::ScalarType> expected_input_types,
    std::span<const std::vector<int64_t>> expected_input_dims,
    std::span<const std::string> expected_input_names,
    std::span<const std::string> expected_output_names)
    -> starpu_server::GrpcModelSpec
{
  return {
      .expected_input_types = expected_input_types,
      .expected_input_dims = expected_input_dims,
      .expected_input_names = expected_input_names,
      .expected_output_names = expected_output_names,
      .max_batch_size = opts.batching.max_batch_size};
}

struct ExpectedModelMetadata {
  std::vector<at::ScalarType> input_types;
  std::vector<std::string> input_names;
  std::vector<std::vector<int64_t>> input_dims;
  std::vector<std::string> output_names;
};

auto
collect_expected_model_metadata(const starpu_server::RuntimeConfig& opts)
    -> ExpectedModelMetadata
{
  ExpectedModelMetadata metadata{};
  if (!opts.model.has_value()) {
    return metadata;
  }

  metadata.input_types.reserve(opts.model->inputs.size());
  metadata.input_names.reserve(opts.model->inputs.size());
  metadata.input_dims.reserve(opts.model->inputs.size());
  for (const auto& input : opts.model->inputs) {
    metadata.input_types.push_back(input.type);
    metadata.input_names.push_back(input.name);
    metadata.input_dims.push_back(input.dims);
  }

  metadata.output_names.reserve(opts.model->outputs.size());
  for (const auto& output : opts.model->outputs) {
    metadata.output_names.push_back(output.name);
  }
  return metadata;
}

void
wait_for_stop_signal(int read_fd, bool use_blocking_signal_wait)
{
  if (use_blocking_signal_wait) {
    if (signal_stop_requested_flag() == 0) {
      wait_for_signal_notification(read_fd);
    }
    return;
  }

  constexpr auto kNotifierSleep = std::chrono::milliseconds(10);
  while (signal_stop_requested_flag() == 0) {
    std::this_thread::sleep_for(kNotifierSleep);
  }
}

auto
make_signal_notifier_thread(
    ServerContext& server_ctx, starpu_server::InferenceQueue& queue,
    ThreadExceptionState& thread_exception_state, int read_fd,
    bool use_blocking_signal_wait) -> std::jthread
{
  return std::jthread([&server_ctx, &queue, &thread_exception_state, read_fd,
                       use_blocking_signal_wait]() {
    run_thread_entry_with_exception_capture(
        "signal-notifier", thread_exception_state, server_ctx, &queue,
        [read_fd, use_blocking_signal_wait, &server_ctx]() {
          wait_for_stop_signal(read_fd, use_blocking_signal_wait);
          if (signal_stop_requested_flag() != 0) {
            request_server_stop(server_ctx);
          }
        });
  });
}

auto
make_worker_thread(
    ServerContext& server_ctx, starpu_server::InferenceQueue& queue,
    ThreadExceptionState& thread_exception_state,
    starpu_server::StarPUTaskRunner& worker) -> std::jthread
{
  return std::jthread(
      [&server_ctx, &queue, &thread_exception_state, &worker]() {
        run_thread_entry_with_exception_capture(
            "starpu-worker", thread_exception_state, server_ctx, &queue,
            [&worker]() { worker.run(); });
      });
}

auto
make_grpc_thread(
    ServerContext& server_ctx, starpu_server::InferenceQueue& queue,
    ThreadExceptionState& thread_exception_state,
    const starpu_server::RuntimeConfig& opts,
    std::vector<torch::Tensor>& reference_outputs,
    const ExpectedModelMetadata& expected_metadata) -> std::jthread
{
  return std::jthread([&server_ctx, &queue, &thread_exception_state, &opts,
                       &reference_outputs, &expected_metadata]() {
    run_thread_entry_with_exception_capture(
        "grpc-server", thread_exception_state, server_ctx, &queue,
        [&]() {
          std::unique_ptr<grpc::Server> grpc_server;
          const auto server_hooks = starpu_server::GrpcServerLifecycleHooks{
              .on_started =
                  [&server_ctx](grpc::Server* server) {
                    mark_server_started(server_ctx, server);
                  },
              .on_stopped =
                  [&server_ctx]() {
                    mark_server_stopped(server_ctx);
                    request_server_stop(server_ctx);
                  }};
          const auto server_options = make_grpc_server_options(opts);
          const auto model_spec = make_grpc_model_spec(
              opts, expected_metadata.input_types, expected_metadata.input_dims,
              expected_metadata.input_names, expected_metadata.output_names);
          starpu_server::RunGrpcServer(
              queue, reference_outputs, model_spec, server_options, grpc_server,
              server_hooks);
        },
        [&server_ctx]() { mark_server_stopped(server_ctx); });
  });
}

void
wait_for_stop_request(ServerContext& server_ctx)
{
  std::unique_lock lock(server_ctx.stop_mutex);
  server_ctx.stop_cv.wait(lock, [&server_ctx] {
    return server_ctx.stop_requested.load(std::memory_order_relaxed);
  });
}

void
log_shutdown_drain_start(
    starpu_server::VerbosityLevel verbosity, std::size_t total_jobs,
    std::size_t completed_before_drain)
{
  if (total_jobs <= completed_before_drain) {
    return;
  }
  const auto remaining_before_drain = total_jobs - completed_before_drain;
  starpu_server::log_info(
      verbosity,
      std::format(
          "Shutdown drain started: completed={} total={} remaining={}.",
          completed_before_drain, total_jobs, remaining_before_drain));
}

void
drain_shutdown_jobs(
    const starpu_server::InferenceQueue& queue, std::size_t total_jobs,
    std::atomic<std::size_t>& completed_jobs,
    std::size_t completed_before_drain, std::condition_variable& all_done_cv,
    std::mutex& all_done_mutex)
{
  if (total_jobs == 0) {
    return;
  }

  notify_shutdown_drain_stage_for_test(
      ShutdownDrainStageForTest::Entered,
      ShutdownDrainProgressForTest{
          .total_jobs = total_jobs,
          .completed_jobs = completed_before_drain,
      });
  const auto shutdown_drain_timeout = resolve_shutdown_drain_timeout();
  const auto shutdown_drain_wait_step = resolve_shutdown_drain_wait_step();
  const auto deadline =
      std::chrono::steady_clock::now() + shutdown_drain_timeout;

  std::unique_lock lock(all_done_mutex);
  while (true) {
    const auto completed = completed_jobs.load(std::memory_order_acquire);
    if (completed >= total_jobs) {
      notify_shutdown_drain_stage_for_test(
          ShutdownDrainStageForTest::CompletedReachedTotal,
          ShutdownDrainProgressForTest{
              .total_jobs = total_jobs,
              .completed_jobs = completed,
          });
      break;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      notify_shutdown_drain_stage_for_test(
          ShutdownDrainStageForTest::DeadlineReached,
          ShutdownDrainProgressForTest{
              .total_jobs = total_jobs,
              .completed_jobs = completed,
          });
      const auto remaining = total_jobs - completed;
      const auto timeout_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              shutdown_drain_timeout)
              .count();
      starpu_server::log_error(std::format(
          "Shutdown drain timeout after {} ms: completed={} total={} "
          "remaining={} queue_size={}.",
          timeout_ms, completed, total_jobs, remaining, queue.size()));
      break;
    }

    const auto until_deadline = deadline - now;
    const auto wait_budget = until_deadline < shutdown_drain_wait_step
                                 ? until_deadline
                                 : shutdown_drain_wait_step;
    notify_shutdown_drain_stage_for_test(
        ShutdownDrainStageForTest::BeforeWait,
        ShutdownDrainProgressForTest{
            .total_jobs = total_jobs,
            .completed_jobs = completed,
        },
        wait_budget);
    static_cast<void>(all_done_cv.wait_for(lock, wait_budget));
  }
}

void
launch_threads(  // NOLINT(readability-function-cognitive-complexity)
    const starpu_server::RuntimeConfig& opts,
    starpu_server::StarPUSetup& starpu, torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    std::vector<torch::Tensor>& reference_outputs,
    starpu_server::InferenceQueue& queue)
{
  queue.reset_counters();
  auto& server_ctx = server_context();
  ThreadExceptionState thread_exception_state;
  server_ctx.stop_requested.store(false, std::memory_order_relaxed);
  signal_stop_requested_flag() = 0;
  reset_server_state(server_ctx);

  SignalNotificationPipe signal_pipe;
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  starpu_server::congestion::start(&queue, make_congestion_config(opts));

  const bool use_blocking_signal_wait = signal_pipe.active();
  std::jthread notifier_thread = make_signal_notifier_thread(
      server_ctx, queue, thread_exception_state, signal_pipe.read_fd(),
      use_blocking_signal_wait);
  struct SignalPipeShutdownGuard {
    SignalNotificationPipe* pipe = nullptr;
    SignalPipeShutdownGuard() = default;
    explicit SignalPipeShutdownGuard(
        SignalNotificationPipe* signal_pipe) noexcept
        : pipe(signal_pipe)
    {
    }
    SignalPipeShutdownGuard(const SignalPipeShutdownGuard&) = delete;
    auto operator=(const SignalPipeShutdownGuard&) -> SignalPipeShutdownGuard& =
                                                          delete;
    SignalPipeShutdownGuard(SignalPipeShutdownGuard&&) = delete;
    auto operator=(SignalPipeShutdownGuard&&) -> SignalPipeShutdownGuard& =
                                                     delete;
    ~SignalPipeShutdownGuard()
    {
      if (pipe != nullptr) {
        pipe->shutdown();
      }
    }
  };
  const SignalPipeShutdownGuard signal_pipe_shutdown_guard{&signal_pipe};

  std::atomic<std::size_t> completed_jobs{0};
  std::condition_variable all_done_cv;
  std::mutex all_done_mutex;

  starpu_server::StarPUTaskRunnerConfig config{};
  config.queue = &queue;
  config.model_cpu = &model_cpu;
  config.models_gpu = &models_gpu;
  config.starpu = &starpu;
  config.opts = &opts;
  config.completed_jobs = &completed_jobs;
  config.all_done_cv = &all_done_cv;
  starpu_server::StarPUTaskRunner worker(config);

  std::jthread worker_thread =
      make_worker_thread(server_ctx, queue, thread_exception_state, worker);
  auto expected_model_metadata = collect_expected_model_metadata(opts);
  std::jthread grpc_thread = make_grpc_thread(
      server_ctx, queue, thread_exception_state, opts, reference_outputs,
      expected_model_metadata);

  wait_for_stop_request(server_ctx);
  stop_server_when_available(server_ctx);
  queue.shutdown();
  const auto total_jobs = queue.total_pushed();
  const auto completed_before_drain =
      completed_jobs.load(std::memory_order_acquire);
  log_shutdown_drain_start(opts.verbosity, total_jobs, completed_before_drain);
  drain_shutdown_jobs(
      queue, total_jobs, completed_jobs, completed_before_drain, all_done_cv,
      all_done_mutex);
  starpu_server::congestion::shutdown();
  server_ctx.stop_cv.notify_one();
  rethrow_thread_exception_if_any(thread_exception_state);
}
