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

STARPU_SERVER_DEFINE_TEST_OVERRIDE_SLOT(
    handle_program_arguments_fatal_override_for_test,
    HandleProgramArgumentsFatalOverrideForTestFn)
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
using LoadModelAndReferenceOutputOverrideForTestFn = std::optional<std::tuple<
    torch::jit::script::Module, std::vector<torch::jit::script::Module>,
    std::vector<torch::Tensor>>> (*)(const starpu_server::RuntimeConfig&);

using RunWarmupOverrideForTestFn = void (*)(
    const starpu_server::RuntimeConfig&, starpu_server::StarPUSetup&,
    torch::jit::script::Module&, std::vector<torch::jit::script::Module>&,
    const std::vector<torch::Tensor>&);

STARPU_SERVER_DEFINE_TEST_OVERRIDE_SLOT(
    load_model_and_reference_output_override_for_test,
    LoadModelAndReferenceOutputOverrideForTestFn)
STARPU_SERVER_DEFINE_TEST_OVERRIDE_SLOT(
    run_warmup_override_for_test, RunWarmupOverrideForTestFn)
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

namespace {

struct LaunchRuntimeState {
  ThreadExceptionState thread_exception_state;
  SignalNotificationPipe signal_pipe;
  std::atomic<std::size_t> completed_jobs{0};
  std::condition_variable all_done_cv;
  std::mutex all_done_mutex;
};

struct RuntimeThreads {
  std::jthread notifier;
  std::jthread worker;
  std::jthread grpc;
};

struct SignalPipeShutdownGuard {
  SignalNotificationPipe* pipe = nullptr;

  SignalPipeShutdownGuard() = default;
  explicit SignalPipeShutdownGuard(SignalNotificationPipe* signal_pipe) noexcept
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

void
start_runtime_threads(
    const starpu_server::RuntimeConfig& opts, ServerContext& server_ctx,
    starpu_server::InferenceQueue& queue,
    std::vector<torch::Tensor>& reference_outputs,
    const ExpectedModelMetadata& expected_model_metadata,
    LaunchRuntimeState& runtime_state, starpu_server::StarPUTaskRunner& worker,
    RuntimeThreads& runtime_threads)
{
  const bool use_blocking_signal_wait = runtime_state.signal_pipe.active();
  runtime_threads.notifier = make_signal_notifier_thread(
      server_ctx, queue, runtime_state.thread_exception_state,
      runtime_state.signal_pipe.read_fd(), use_blocking_signal_wait);
  runtime_threads.worker = make_worker_thread(
      server_ctx, queue, runtime_state.thread_exception_state, worker);
  runtime_threads.grpc = make_grpc_thread(
      server_ctx, queue, runtime_state.thread_exception_state, opts,
      reference_outputs, expected_model_metadata);
}

}  // namespace

void
launch_threads(
    const starpu_server::RuntimeConfig& opts,
    starpu_server::StarPUSetup& starpu, torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    std::vector<torch::Tensor>& reference_outputs,
    starpu_server::InferenceQueue& queue)
{
  auto& server_ctx = server_context();
  LaunchRuntimeState runtime_state;
  setup_runtime_state(opts, server_ctx, queue);

  starpu_server::StarPUTaskRunnerConfig config{};
  config.queue = &queue;
  config.model_cpu = &model_cpu;
  config.models_gpu = &models_gpu;
  config.starpu = &starpu;
  config.opts = &opts;
  config.completed_jobs = &runtime_state.completed_jobs;
  config.all_done_cv = &runtime_state.all_done_cv;
  starpu_server::StarPUTaskRunner worker(config);
  auto expected_model_metadata = collect_expected_model_metadata(opts);

  RuntimeThreads runtime_threads;
  const SignalPipeShutdownGuard signal_pipe_shutdown_guard{
      &runtime_state.signal_pipe};
  start_runtime_threads(
      opts, server_ctx, queue, reference_outputs, expected_model_metadata,
      runtime_state, worker, runtime_threads);
  const ShutdownRuntimeContext shutdown_context{
      .thread_exception_state = &runtime_state.thread_exception_state,
      .completed_jobs = &runtime_state.completed_jobs,
      .all_done_cv = &runtime_state.all_done_cv,
      .all_done_mutex = &runtime_state.all_done_mutex};
  run_shutdown_sequence(opts, server_ctx, queue, shutdown_context);
}
