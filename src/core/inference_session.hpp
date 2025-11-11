#pragma once

#include <torch/script.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "inference_queue.hpp"
#include "inference_runner.hpp"
#include "starpu_task_worker/starpu_task_worker.hpp"

namespace starpu_server {

struct RuntimeConfig;
class StarPUSetup;

class InferenceSession {
 public:
  using ClientRoutine = std::function<void(
      InferenceQueue&, const RuntimeConfig&, const std::vector<torch::Tensor>&,
      int request_nb)>;

  InferenceSession(
      const RuntimeConfig& opts, StarPUSetup& starpu,
      ClientRoutine client_routine = nullptr);
  ~InferenceSession();

  void run();

 private:
  auto load_models_and_reference() -> bool;
  void warmup();
  void prepare_results_storage();
  void configure_worker();
  void launch_threads();
  void launch_client_thread();
  void await_completion();
  void join_threads();
  void report_latency_stats() const;
  void process_results();

  const RuntimeConfig& opts_;
  StarPUSetup& starpu_;
  ClientRoutine client_routine_;

  torch::jit::script::Module model_cpu_;
  std::vector<torch::jit::script::Module> models_gpu_;
  std::vector<torch::Tensor> outputs_ref_;

  InferenceQueue queue_;
  std::vector<InferenceResult> results_;
  std::mutex results_mutex_;

  std::atomic<int> completed_jobs_{0};
  std::condition_variable all_done_cv_;
  std::mutex all_done_mutex_;

  StarPUTaskRunnerConfig config_{};
  std::unique_ptr<StarPUTaskRunner> worker_;

  std::jthread server_thread_;
  std::jthread client_thread_;
};

}  // namespace starpu_server
