inline auto
select_earliest_time(Clock::time_point current, Clock::time_point candidate)
    -> Clock::time_point
{
  if (candidate == Clock::time_point{}) {
    return current;
  }
  if (current == Clock::time_point{} || candidate < current) {
    return candidate;
  }
  return current;
}

inline auto
select_latest_time(Clock::time_point current, Clock::time_point candidate)
    -> Clock::time_point
{
  if (candidate == Clock::time_point{}) {
    return current;
  }
  if (current == Clock::time_point{} || candidate > current) {
    return candidate;
  }
  return current;
}

auto
slice_outputs_for_sub_job(
    const std::vector<torch::Tensor>& aggregated_outputs,
    SubJobSliceOptions options) -> SubJobSliceResult
{
  SubJobSliceResult result;
  const int64_t slice_size = std::max<int64_t>(1, options.batch_size);
  result.processed_length = slice_size;

  if (aggregated_outputs.empty()) {
    return result;
  }

  result.outputs.reserve(aggregated_outputs.size());
  bool determined_length = false;
  const auto slice_start = static_cast<int64_t>(options.offset);

  for (const auto& tensor : aggregated_outputs) {
    if (!tensor.defined() || tensor.dim() == 0) {
      result.outputs.push_back(tensor);
      continue;
    }

    const int64_t available = tensor.size(0);
    const int64_t slice_end =
        std::min<int64_t>(available, slice_start + slice_size);
    const int64_t length = std::max<int64_t>(0, slice_end - slice_start);

    if (length <= 0) {
      result.outputs.emplace_back();
      continue;
    }

    if (!determined_length) {
      result.processed_length = length;
      determined_length = true;
    }

    auto slice_view = tensor.narrow(0, slice_start, length);
    if (!slice_view.is_contiguous()) {
      slice_view = slice_view.contiguous();
    }
    result.outputs.push_back(std::move(slice_view));
  }

  return result;
}

auto
aggregate_batch_metadata(
    const std::vector<std::shared_ptr<InferenceJob>>& jobs,
    const RuntimeConfig* opts) -> BatchAggregationInfo
{
  BatchAggregationInfo info;
  if (jobs.empty()) {
    return info;
  }

  info.sub_jobs.reserve(jobs.size());
  info.earliest_start = jobs.front()->get_start_time();
  info.earliest_enqueued = jobs.front()->timing_info().enqueued_time;
  info.latest_enqueued = jobs.front()->timing_info().enqueued_time;
  info.earliest_batch_collect_start =
      jobs.front()->timing_info().batch_collect_start_time;

  for (const auto& job : jobs) {
    const auto job_batch = resolve_batch_size_for_job(opts, job);
    info.total_samples += job_batch > 0 ? job_batch : 1;
    info.logical_jobs += std::max(1, job->logical_job_count());
    info.earliest_start =
        select_earliest_time(info.earliest_start, job->get_start_time());
    info.earliest_enqueued = select_earliest_time(
        info.earliest_enqueued, job->timing_info().enqueued_time);
    info.latest_enqueued = select_latest_time(
        info.latest_enqueued, job->timing_info().enqueued_time);
    info.earliest_batch_collect_start = select_earliest_time(
        info.earliest_batch_collect_start,
        job->timing_info().batch_collect_start_time);

    InferenceJob::AggregatedSubJob entry{};
    entry.job = std::weak_ptr<InferenceJob>(job);
    entry.callback = job->get_on_complete();
    entry.batch_size = job_batch;
    entry.request_id = job->get_request_id();
    entry.arrival_time = job->timing_info().enqueued_time;
    info.sub_jobs.push_back(std::move(entry));
  }

  return info;
}

auto
resize_outputs_for_batch(
    const std::vector<torch::Tensor>& prototype_outputs,
    int64_t batch_size) -> std::vector<torch::Tensor>
{
  std::vector<torch::Tensor> resized;
  resized.reserve(prototype_outputs.size());
  for (const auto& out : prototype_outputs) {
    if (!out.defined()) {
      resized.emplace_back(out);
      continue;
    }
    std::vector<int64_t> shape(out.sizes().begin(), out.sizes().end());
    if (!shape.empty()) {
      shape.front() = batch_size;
    }
    resized.emplace_back(torch::empty(shape, out.options()));
  }
  return resized;
}

void
release_inputs_from_additional_jobs(
    std::vector<std::shared_ptr<InferenceJob>>& jobs)
{
  for (size_t idx = 1; idx < jobs.size(); ++idx) {
    if (!jobs[idx]) {
      continue;
    }
    static_cast<void>(jobs[idx]->release_input_tensors());
    jobs[idx]->set_input_memory_holders(
        std::vector<std::shared_ptr<const void>>{});
  }
}

[[nodiscard]] auto
request_id_from_sub_job(const InferenceJob::AggregatedSubJob& sub_job) -> int
{
  if (sub_job.request_id >= 0) {
    return sub_job.request_id;
  }
  if (auto locked = sub_job.job.lock()) {
    return locked->get_request_id();
  }
  return sub_job.request_id;
}

auto
build_request_ids_for_trace(const std::shared_ptr<InferenceJob>& job)
    -> std::vector<int>
{
  if (!job) {
    return {};
  }

  if (!job->has_aggregated_sub_jobs()) {
    return std::vector<int>{job->get_request_id()};
  }

  const auto& aggregated = job->aggregated_sub_jobs();
  std::vector<int> ids;
  ids.reserve(aggregated.size());
  for (const auto& sub_job : aggregated) {
    ids.push_back(request_id_from_sub_job(sub_job));
  }

  return ids;
}

auto
build_request_arrival_us_for_trace(const std::shared_ptr<InferenceJob>& job)
    -> std::vector<int64_t>
{
  const auto to_microseconds =
      [](MonotonicClock::time_point time_point) -> int64_t {
    if (time_point == MonotonicClock::time_point{}) {
      return 0;
    }
    return std::chrono::duration_cast<std::chrono::microseconds>(
               time_point.time_since_epoch())
        .count();
  };

  if (!job) {
    return {};
  }

  if (!job->has_aggregated_sub_jobs()) {
    return std::vector<int64_t>{
        to_microseconds(job->timing_info().enqueued_time)};
  }

  const auto& aggregated = job->aggregated_sub_jobs();
  std::vector<int64_t> arrivals;
  arrivals.reserve(aggregated.size());
  for (const auto& sub_job : aggregated) {
    auto arrival = sub_job.arrival_time;
    if (arrival == MonotonicClock::time_point{}) {
      if (auto locked = sub_job.job.lock()) {
        arrival = locked->timing_info().enqueued_time;
      }
    }
    arrivals.push_back(to_microseconds(arrival));
  }
  return arrivals;
}
