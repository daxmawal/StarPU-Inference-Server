#pragma once

#include <torch/script.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

#include "Inference_queue.hpp"
#include "args_parser.hpp"
#include "starpu_setup.hpp"

// ============================================================================
// ServerWorker
// ----------------------------------------------------------------------------
// Threaded worker responsible for:
//  - Pulling jobs from the inference queue
//  - Submitting them to StarPU
//  - Collecting and storing results
// ============================================================================
class ServerWorker {
 public:
  ServerWorker(
      InferenceQueue* queue, torch::jit::script::Module* model_cpu,
      std::vector<torch::jit::script::Module>* models_gpu, StarPUSetup* starpu,
      const ProgramOptions* opts, std::vector<InferenceResult>* results,
      std::mutex* results_mutex, std::atomic<unsigned int>* completed_jobs,
      std::condition_variable* all_done_cv);

  /// Main job-processing loop
  void run();

 private:
  // Input
  InferenceQueue* queue_;
  torch::jit::script::Module* model_cpu_;
  std::vector<torch::jit::script::Module>* models_gpu_;
  StarPUSetup* starpu_;
  const ProgramOptions* opts_;

  // Output and synchronization
  std::vector<InferenceResult>* results_;
  std::mutex* results_mutex_;
  std::atomic<unsigned int>* completed_jobs_;
  std::condition_variable* all_done_cv_;
};
