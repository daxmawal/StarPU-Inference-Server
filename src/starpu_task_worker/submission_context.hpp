#ifndef STARPU_INFERENCE_SERVER_STARPU_TASK_WORKER_SUBMISSION_CONTEXT_HPP_
#define STARPU_INFERENCE_SERVER_STARPU_TASK_WORKER_SUBMISSION_CONTEXT_HPP_

namespace starpu_server {

enum class SubmissionPhase { Probe, RealInference };

class SubmissionPhaseContext {
 private:
  static thread_local SubmissionPhase& get_phase_ref()
  {
    static thread_local SubmissionPhase phase = SubmissionPhase::RealInference;
    return phase;
  }

 public:
  static auto current_phase() -> SubmissionPhase { return get_phase_ref(); }

  static void set_phase(SubmissionPhase phase) { get_phase_ref() = phase; }

 private:
  SubmissionPhaseContext() = default;
};

class SubmissionPhaseScopedGuard {
 public:
  explicit SubmissionPhaseScopedGuard(SubmissionPhase phase)
      : previous_(SubmissionPhaseContext::current_phase())
  {
    SubmissionPhaseContext::set_phase(phase);
  }

  SubmissionPhaseScopedGuard(const SubmissionPhaseScopedGuard&) = delete;
  auto operator=(const SubmissionPhaseScopedGuard&)
      -> SubmissionPhaseScopedGuard& = delete;

  ~SubmissionPhaseScopedGuard()
  {
    SubmissionPhaseContext::set_phase(previous_);
  }

 private:
  SubmissionPhase previous_;
};

}  // namespace starpu_server

#endif  // STARPU_INFERENCE_SERVER_STARPU_TASK_WORKER_SUBMISSION_CONTEXT_HPP_
