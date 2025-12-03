#include "inference_session.hpp"

#include <format>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "inference_runner.hpp"
#include "utils/logger.hpp"

namespace starpu_server {

InferenceSession::InferenceSession(
    const RuntimeConfig& opts, StarPUSetup& starpu,
    ClientRoutine client_routine)
    : opts_(opts), starpu_(starpu), client_routine_(std::move(client_routine))
{
}

InferenceSession::~InferenceSession()
{
  join_threads();
}

void
InferenceSession::run()
{
  if (!load_models_and_reference()) {
    return;
  }

  warmup();
  configure_worker();
  launch_threads();
  await_completion();
  join_threads();
}

auto
InferenceSession::load_models_and_reference() -> bool
{
  try {
    auto result = load_model_and_reference_output(opts_);
    if (!result) {
      log_error("Failed to load model or reference outputs");
      return false;
    }
    std::tie(model_cpu_, models_gpu_, outputs_ref_) = std::move(*result);
    return true;
  }
  catch (const InferenceEngineException& e) {
    log_error(
        std::format("Failed to load model or reference outputs: {}", e.what()));
  }
  return false;
}

void
InferenceSession::warmup()
{
  run_warmup(opts_, starpu_, model_cpu_, models_gpu_, outputs_ref_);
}

void
InferenceSession::configure_worker()
{
  config_.queue = &queue_;
  config_.model_cpu = &model_cpu_;
  config_.models_gpu = &models_gpu_;
  config_.starpu = &starpu_;
  config_.opts = &opts_;
  config_.completed_jobs = &completed_jobs_;
  config_.all_done_cv = &all_done_cv_;
  worker_ = std::make_unique<StarPUTaskRunner>(config_);
}

void
InferenceSession::launch_threads()
{
  if (!worker_) {
    throw std::logic_error("InferenceSession worker not configured");
  }
  try {
    server_thread_ = get_worker_thread_launcher()(*worker_);
    if (client_routine_) {
      launch_client_thread();
    }
  }
  catch (const std::exception& e) {
    log_error(std::format("Failed to start worker thread: {}", e.what()));
    queue_.shutdown();
    join_threads();
    throw;
  }
}

void
InferenceSession::launch_client_thread()
{
  client_thread_ = std::jthread([this]() {
    client_routine_(queue_, opts_, outputs_ref_, opts_.batching.request_nb);
  });
}

void
InferenceSession::await_completion()
{
  std::unique_lock lock(all_done_mutex_);
  all_done_cv_.wait(lock, [this]() {
    return completed_jobs_.load(std::memory_order_acquire) >=
           opts_.batching.request_nb;
  });
}

void
InferenceSession::join_threads()
{
  if (client_thread_.joinable()) {
    client_thread_.join();
  }
  if (server_thread_.joinable()) {
    server_thread_.join();
  }
}

}  // namespace starpu_server
