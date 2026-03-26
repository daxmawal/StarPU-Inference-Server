void
InferenceServiceImpl::validate_schema_or_throw() const
{
  for (std::size_t i = 0; i < expected_input_types_.size(); ++i) {
    const at::ScalarType dtype = expected_input_types_[i];
    if (scalar_type_to_model_dtype(dtype) ==
        inference::DataType::TYPE_INVALID) {
      throw std::invalid_argument(std::format(
          "Invalid schema: unsupported input datatype at index {}", i));
    }
    (void)scalar_type_to_datatype(dtype);
  }

  if (reference_outputs_ == nullptr) {
    return;
  }

  for (std::size_t i = 0; i < reference_outputs_->size(); ++i) {
    const at::ScalarType dtype = (*reference_outputs_)[i].scalar_type();
    if (scalar_type_to_model_dtype(dtype) ==
        inference::DataType::TYPE_INVALID) {
      throw std::invalid_argument(std::format(
          "Invalid schema: unsupported output datatype at index {}", i));
    }
    (void)scalar_type_to_datatype(dtype);
  }
}

auto
InferenceServiceImpl::validate_and_convert_inputs(
    const ModelInferRequest* request, std::vector<torch::Tensor>& inputs,
    std::vector<std::shared_ptr<const void>>* input_lifetimes) -> Status
{
  if (auto status =
          validate_input_counts(request, expected_input_types_.size());
      !status.ok()) {
    return status;
  }

  const auto name_state = collect_input_name_state(request);
  const bool has_expected_names = !expected_input_names_.empty();
  if (auto status = validate_input_names(name_state, has_expected_names);
      !status.ok()) {
    return status;
  }
  const bool use_name_mapping = has_expected_names;

  std::unordered_map<std::string_view, std::size_t> expected_index_by_name;
  if (use_name_mapping) {
    if (auto status = build_expected_index_by_name(
            expected_input_names_, expected_index_by_name);
        !status.ok()) {
      return status;
    }
  }

  const std::size_t expected_count = expected_input_types_.size();
  std::vector<torch::Tensor> ordered_inputs(expected_count);
  std::vector<std::shared_ptr<const void>> ordered_lifetimes;
  if (input_lifetimes != nullptr) {
    ordered_lifetimes.resize(expected_count);
  }
  std::vector<bool> filled(expected_count, false);
  const ProcessInputContext process_context{
      use_name_mapping ? &expected_index_by_name : nullptr,
      &expected_input_types_,
      &expected_input_dims_,
      max_batch_size_,
      &ordered_inputs,
      input_lifetimes != nullptr ? &ordered_lifetimes : nullptr,
      &filled};

  for (int i = 0; i < request->inputs_size(); ++i) {
    Status status = process_input(request, i, process_context);
    if (!status.ok()) {
      return status;
    }
  }

  if (use_name_mapping) {
    Status status = check_missing_named_inputs(filled, expected_input_names_);
    if (!status.ok()) {
      return status;
    }
  }

  inputs = std::move(ordered_inputs);
  if (input_lifetimes != nullptr) {
    *input_lifetimes = std::move(ordered_lifetimes);
  }
  return Status::OK;
}

auto
InferenceServiceImpl::populate_response(
    const ModelInferRequest* request, ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs, int64_t recv_ms,
    const LatencyBreakdown& breakdown) -> Status
{
  return populate_response(
      request, reply, outputs, recv_ms, breakdown, PopulateResponseOptions{});
}

auto
InferenceServiceImpl::populate_response(
    const ModelInferRequest* request, ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs, int64_t recv_ms,
    const LatencyBreakdown& breakdown,
    PopulateResponseOptions options) -> Status
{
  if (request == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "ModelInfer request is null"};
  }
  if (reply == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "ModelInfer response is null"};
  }
  if (!options.model_name_override.empty()) {
    reply->set_model_name(std::string(options.model_name_override));
  } else {
    reply->set_model_name(request->model_name());
  }
  reply->set_model_version(request->model_version());
  reply->set_server_receive_ms(recv_ms);
  reply->set_server_queue_ms(breakdown.queue_ms);
  reply->set_server_batch_ms(breakdown.batch_ms);
  reply->set_server_submit_ms(breakdown.submit_ms);
  reply->set_server_scheduling_ms(breakdown.scheduling_ms);
  reply->set_server_codelet_ms(breakdown.codelet_ms);
  reply->set_server_inference_ms(breakdown.inference_ms);
  reply->set_server_callback_ms(breakdown.callback_ms);
  if (options.set_prepost_overall) {
    reply->set_server_preprocess_ms(breakdown.preprocess_ms);
    reply->set_server_postprocess_ms(breakdown.postprocess_ms);
    reply->set_server_overall_ms(breakdown.overall_ms);
  }
  reply->set_server_total_ms(breakdown.total_ms);

  const auto resolved_names =
      resolve_output_names(options.output_names, outputs.size());
  std::vector<std::size_t> output_indices;
  if (request->outputs_size() > 0) {
    std::unordered_map<std::string_view, std::size_t> index_by_name;
    index_by_name.reserve(resolved_names.size());
    for (std::size_t i = 0; i < resolved_names.size(); ++i) {
      const auto [existing_it, inserted] =
          index_by_name.try_emplace(resolved_names[i], i);
      if (!inserted) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format(
                "Configured output name '{}' is duplicated",
                existing_it->first)};
      }
    }

    std::unordered_set<std::string_view> seen;
    output_indices.reserve(request->outputs_size());
    for (const auto& requested : request->outputs()) {
      if (requested.name().empty()) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            "Requested output name must be non-empty"};
      }
      const auto name_iter = index_by_name.find(requested.name());
      if (name_iter == index_by_name.end()) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format(
                "Requested output '{}' is not available", requested.name())};
      }
      if (!seen.insert(name_iter->first).second) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            std::format(
                "Requested output '{}' is duplicated", requested.name())};
      }
      output_indices.push_back(name_iter->second);
    }
  } else {
    output_indices.reserve(outputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      output_indices.push_back(i);
    }
  }

  return fill_output_tensor(reply, outputs, output_indices, resolved_names);
}
