auto
signal_stop_requested_flag() -> volatile std::sig_atomic_t&
{
  static volatile std::sig_atomic_t value = 0;
  return value;
}

auto
signal_stop_notify_fd() -> volatile std::sig_atomic_t&
{
  static volatile std::sig_atomic_t value = -1;
  return value;
}

class SignalNotificationPipe {
 public:
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  static void SetPipeFailureForTest(bool enabled) noexcept
  {
    pipe_failure_for_test() = enabled;
  }

  static void SetSetNonBlockingFailureForTest(bool enabled) noexcept
  {
    set_non_blocking_failure_for_test() = enabled;
  }

  static auto SetNonBlockingForTest(int file_descriptor) -> bool
  {
    return set_non_blocking(file_descriptor);
  }
#endif  // SONAR_IGNORE_STOP

  SignalNotificationPipe()
  {
    std::array<int, 2> file_descriptors{-1, -1};
    if (create_pipe(file_descriptors) != 0) {
      starpu_server::log_warning(std::format(
          "Failed to create stop-notification pipe; falling back to polling "
          "signal flag: {}",
          std::strerror(errno)));
      return;
    }
    read_fd_ = file_descriptors[0];
    write_fd_ = file_descriptors[1];
    if (!set_non_blocking(write_fd_)) {
      starpu_server::log_warning(std::format(
          "Failed to configure stop-notification pipe write end as "
          "non-blocking; falling back to polling signal flag: {}",
          std::strerror(errno)));
      close_fd_noexcept(read_fd_);
      close_fd_noexcept(write_fd_);
      read_fd_ = -1;
      write_fd_ = -1;
      return;
    }
    signal_stop_notify_fd() = static_cast<std::sig_atomic_t>(write_fd_);
  }

  SignalNotificationPipe(const SignalNotificationPipe&) = delete;
  auto operator=(const SignalNotificationPipe&) -> SignalNotificationPipe& =
                                                       delete;
  SignalNotificationPipe(SignalNotificationPipe&&) = delete;
  auto operator=(SignalNotificationPipe&&) -> SignalNotificationPipe& = delete;

  ~SignalNotificationPipe() noexcept
  {
    shutdown();
    close_fd_noexcept(read_fd_);
  }

  void shutdown() noexcept
  {
    if (write_fd_ >= 0) {
      signal_stop_notify_fd() = -1;
      close_fd_noexcept(write_fd_);
      write_fd_ = -1;
    } else {
      signal_stop_notify_fd() = -1;
    }
  }

  [[nodiscard]] auto read_fd() const noexcept -> int { return read_fd_; }

  [[nodiscard]] auto active() const noexcept -> bool
  {
    return read_fd_ >= 0 && write_fd_ >= 0;
  }

 private:
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  static auto pipe_failure_for_test() noexcept -> bool&
  {
    static bool enabled = false;
    return enabled;
  }

  static auto set_non_blocking_failure_for_test() noexcept -> bool&
  {
    static bool enabled = false;
    return enabled;
  }
#endif  // SONAR_IGNORE_STOP

  static auto create_pipe(std::array<int, 2>& file_descriptors) -> int
  {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (pipe_failure_for_test()) {
      errno = EMFILE;
      return -1;
    }
#endif  // SONAR_IGNORE_STOP
    return ::pipe(file_descriptors.data());
  }

  static auto set_non_blocking(int file_descriptor) -> bool
  {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (set_non_blocking_failure_for_test()) {
      errno = EINVAL;
      return false;
    }
#endif  // SONAR_IGNORE_STOP
    if (file_descriptor < 0) {
      return false;
    }
    // POSIX `fcntl` is variadic by API contract.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    const int flags = ::fcntl(file_descriptor, F_GETFL);
    if (flags < 0) {
      return false;
    }
    // POSIX `fcntl` is variadic by API contract.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    return ::fcntl(file_descriptor, F_SETFL, flags | O_NONBLOCK) == 0;
  }

  static void close_fd_noexcept(int file_descriptor) noexcept
  {
    if (file_descriptor >= 0) {
      (void)::close(file_descriptor);
    }
  }

  int read_fd_ = -1;
  int write_fd_ = -1;
};

void
request_server_stop(ServerContext& ctx)
{
  ctx.stop_requested.store(true, std::memory_order_relaxed);
  ctx.stop_cv.notify_one();
}

struct NoopExceptionHook {
  void operator()() const noexcept {}
};

template <typename Fn, typename OnExceptionFn = NoopExceptionHook>
void
run_thread_entry_with_exception_capture(
    std::string_view thread_name, ThreadExceptionState& state,
    ServerContext& server_ctx, starpu_server::InferenceQueue* queue,
    Fn&& thread_entry, OnExceptionFn&& on_exception = {}) noexcept
{
  auto invoke_on_exception = [&on_exception, thread_name]() {
    try {
      on_exception();
    }
    catch (const std::exception& hook_exception) {
      starpu_server::log_error(std::format(
          "Unhandled exception escaped '{}' exception hook: {}", thread_name,
          hook_exception.what()));
    }
    catch (...) {
      starpu_server::log_error(std::format(
          "Unhandled non-standard exception escaped '{}' exception hook.",
          thread_name));
    }
  };

  try {
    std::forward<Fn>(thread_entry)();
  }
  catch (const std::exception& e) {
    starpu_server::log_error(std::format(
        "Unhandled exception escaped '{}' thread: {}", thread_name, e.what()));
    state.capture(thread_name, std::current_exception());
    if (queue != nullptr) {
      queue->shutdown();
    }
    invoke_on_exception();
    request_server_stop(server_ctx);
  }
  catch (...) {
    starpu_server::log_error(std::format(
        "Unhandled non-standard exception escaped '{}' thread.", thread_name));
    state.capture(thread_name, std::current_exception());
    if (queue != nullptr) {
      queue->shutdown();
    }
    invoke_on_exception();
    request_server_stop(server_ctx);
  }
}

void
rethrow_thread_exception_if_any(ThreadExceptionState& state)
{
  auto [thread_exception, thread_name] = state.take();
  if (thread_exception == nullptr) {
    return;
  }

  try {
    std::rethrow_exception(thread_exception);
  }
  catch (const std::exception& e) {
    throw starpu_server::WorkerThreadException(std::format(
        "Thread '{}' terminated with exception: {}", thread_name, e.what()));
  }
  catch (...) {
    throw starpu_server::WorkerThreadException(std::format(
        "Thread '{}' terminated with unknown exception.", thread_name));
  }
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using WaitForSignalNotificationReadOverrideForTestFn =
    ssize_t (*)(int, void*, std::size_t);

auto
wait_for_signal_notification_read_override_for_test() noexcept
    -> WaitForSignalNotificationReadOverrideForTestFn&
{
  static WaitForSignalNotificationReadOverrideForTestFn override_fn = nullptr;
  return override_fn;
}
#endif  // SONAR_IGNORE_STOP

auto
read_signal_notification(int read_fd, void* buffer, std::size_t buffer_size)
    -> ssize_t
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn =
          wait_for_signal_notification_read_override_for_test();
      override_fn != nullptr) {
    return override_fn(read_fd, buffer, buffer_size);
  }
#endif  // SONAR_IGNORE_STOP
  return ::read(read_fd, buffer, buffer_size);
}

void
wait_for_signal_notification(int read_fd)
{
  if (read_fd < 0) {
    return;
  }
  constexpr std::size_t kSignalNotificationBufferSize = 16;
  std::array<char, kSignalNotificationBufferSize> buffer{};
  while (true) {
    const ssize_t bytes_read =
        read_signal_notification(read_fd, buffer.data(), buffer.size());
    if (bytes_read > 0 || bytes_read == 0) {
      return;
    }
    if (errno == EINTR) {
      continue;
    }
    starpu_server::log_warning(std::format(
        "Failed while waiting for stop signal notification: {}",
        std::strerror(errno)));
    return;
  }
}

auto
server_context() -> ServerContext&
{
  static ServerContext ctx;
  return ctx;
}

void
reset_server_state(ServerContext& ctx)
{
  std::lock_guard lock(ctx.server_mutex);
  ctx.server = nullptr;
  ctx.server_startup_observed = false;
}

void
mark_server_started(ServerContext& ctx, grpc::Server* server)
{
  {
    std::lock_guard lock(ctx.server_mutex);
    ctx.server = server;
    ctx.server_startup_observed = true;
  }
  ctx.server_cv.notify_all();
}

void
mark_server_stopped(ServerContext& ctx)
{
  {
    std::lock_guard lock(ctx.server_mutex);
    ctx.server = nullptr;
    ctx.server_startup_observed = true;
  }
  ctx.server_cv.notify_all();
}

void
stop_server_when_available(ServerContext& ctx)
{
  std::unique_lock lock(ctx.server_mutex);
  ctx.server_cv.wait(lock, [&ctx]() { return ctx.server_startup_observed; });
  starpu_server::StopServer(ctx.server);
}
