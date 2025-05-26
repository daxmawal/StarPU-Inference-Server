#pragma once

#include <torch/script.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

#include "Inference_queue.hpp"
#include "args_parser.hpp"
#include "starpu_setup.hpp"

class ServerWorker {
 public:
  ServerWorker(
      InferenceQueue& queue, torch::jit::script::Module& model_cpu,
      torch::jit::script::Module& model_gpu, StarPUSetup& starpu,
      const ProgramOptions& opts, std::vector<InferenceResult>& results,
      std::mutex& results_mutex, std::atomic<unsigned int>& completed_jobs,
      std::condition_variable& all_done_cv)
      : queue_(queue), model_cpu_(model_cpu), model_gpu_(model_gpu),
        starpu_(starpu), opts_(opts), results_(results),
        results_mutex_(results_mutex), completed_jobs_(completed_jobs),
        all_done_cv_(all_done_cv)
  {
  }

  void run();

 private:
  InferenceQueue& queue_;
  torch::jit::script::Module& model_cpu_;
  torch::jit::script::Module& model_gpu_;
  StarPUSetup& starpu_;
  const ProgramOptions& opts_;
  std::vector<InferenceResult>& results_;
  std::mutex& results_mutex_;
  std::atomic<unsigned int>& completed_jobs_;
  std::condition_variable& all_done_cv_;
};