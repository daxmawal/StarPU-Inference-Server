struct StarPUTaskRunner::PreparedJobProcessingContext {
  std::shared_ptr<InferenceJob> job;
  int job_id = -1;
  int request_id = -1;
  int logical_jobs = 0;
  int submission_id = -1;
  bool warmup_job = false;
};

void
StarPUTaskRunner::process_prepared_job(const std::shared_ptr<InferenceJob>& job)
{
  PreparedJobProcessingContext context{};
  context.job = job;
  if (context.job == nullptr) {
    return;
  }

  context.job_id = context.job->get_request_id();
  try {
    if (context.job->is_cancelled()) {
      handle_cancelled_job(context.job);
      return;
    }

    context.submission_id = next_submission_id_.fetch_add(1);
    context.job->set_submission_id(context.submission_id);
    context.job->update_timing_info(
        [submission_id = context.submission_id](detail::TimingInfo& timing) {
          timing.submission_id = submission_id;
        });

    context.logical_jobs = context.job->logical_job_count();
    context.request_id = context.job->get_request_id();
    context.job_id = task_runner_internal::job_identifier(*context.job);
    if (should_log(VerbosityLevel::Trace, opts_->verbosity)) {
      log_trace(
          opts_->verbosity,
          std::format(
              "Dequeued job submission {} (request {}), queue size : {}, "
              "aggregated requests: {}",
              context.job_id, context.request_id, queue_->size(),
              context.logical_jobs));
    }

    context.warmup_job = is_warmup_job(context.job);
    trace_batch_if_enabled(
        context.job, context.warmup_job, context.submission_id);
    prepare_job_completion_callback(context.job);
    task_runner_internal::invoke_run_before_submit_hook();
    submit_job_or_handle_failure(
        context.job, SubmissionInfo{context.submission_id, context.job_id});
  }
  catch (const std::exception& exception) {
    finalize_job_after_exception(
        context.job, exception,
        "Unexpected exception while processing dequeued job", context.job_id);
  }
  catch (...) {
    finalize_job_after_unknown_exception(
        context.job,
        "Unexpected non-standard exception while processing dequeued job",
        context.job_id);
  }
}
