namespace {
template <typename AddEntry, typename SetDataType, typename AppendDim>
auto
append_input_schema(
    std::span<const at::ScalarType> expected_input_types,
    std::span<const std::vector<int64_t>> expected_input_dims,
    std::span<const std::string> expected_input_names, AddEntry&& add_entry,
    SetDataType&& set_data_type, AppendDim&& append_dim) -> Status
{
  auto add_entry_fn = std::forward<AddEntry>(add_entry);
  auto set_data_type_fn = std::forward<SetDataType>(set_data_type);
  auto append_dim_fn = std::forward<AppendDim>(append_dim);

  for (std::size_t i = 0; i < expected_input_types.size(); ++i) {
    auto* input = add_entry_fn();
    input->set_name(resolve_tensor_name(i, expected_input_names, "input"));
    if (const Status status = set_data_type_fn(*input, expected_input_types[i]);
        !status.ok()) {
      return status;
    }
    if (i < expected_input_dims.size()) {
      for (const auto dim : expected_input_dims[i]) {
        append_dim_fn(*input, dim);
      }
    }
  }
  return Status::OK;
}

template <typename AddEntry, typename SetDataType, typename AppendDim>
auto
append_output_schema(
    const std::vector<torch::Tensor>* reference_outputs,
    std::span<const std::string> expected_output_names, AddEntry&& add_entry,
    SetDataType&& set_data_type, AppendDim&& append_dim) -> Status
{
  auto add_entry_fn = std::forward<AddEntry>(add_entry);
  auto set_data_type_fn = std::forward<SetDataType>(set_data_type);
  auto append_dim_fn = std::forward<AppendDim>(append_dim);

  if (reference_outputs == nullptr) {
    return Status::OK;
  }

  for (std::size_t i = 0; i < reference_outputs->size(); ++i) {
    const auto& output = (*reference_outputs)[i];
    auto* output_entry = add_entry_fn();
    output_entry->set_name(
        resolve_tensor_name(i, expected_output_names, "output"));
    if (const Status status =
            set_data_type_fn(*output_entry, output.scalar_type());
        !status.ok()) {
      return status;
    }
    for (const auto dim : output.sizes()) {
      append_dim_fn(*output_entry, dim);
    }
  }

  return Status::OK;
}
}  // namespace

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
InferenceServiceImpl::ServerLive(
    ServerContext* /*context*/, const ServerLiveRequest* /*request*/,
    ServerLiveResponse* reply) -> Status
{
  reply->set_live(true);
  return Status::OK;
}

auto
InferenceServiceImpl::ServerReady(
    ServerContext* /*context*/, const ServerReadyRequest* /*request*/,
    ServerReadyResponse* reply) -> Status
{
  const bool ready = queue_ != nullptr && !queue_->is_shutdown();
  reply->set_ready(ready);
  return Status::OK;
}

auto
InferenceServiceImpl::ModelReady(
    ServerContext* /*context*/, const ModelReadyRequest* request,
    ModelReadyResponse* reply) -> Status
{
  bool ready = queue_ != nullptr && !queue_->is_shutdown();
  if (ready && !default_model_name_.empty()) {
    const auto& requested_name = request->name();
    if (!requested_name.empty() && requested_name != default_model_name_) {
      ready = false;
    }
  }
  reply->set_ready(ready);
  return Status::OK;
}

auto
InferenceServiceImpl::ServerMetadata(
    ServerContext* /*context*/,
    const inference::ServerMetadataRequest* /*request*/,
    inference::ServerMetadataResponse* reply) -> Status
{
  std::string name = server_name_;
  if (name.empty()) {
    name = default_model_name_;
  }
  if (name.empty()) {
    name = "starpu_server";
  }
  reply->set_name(std::move(name));
  if (!server_version_.empty()) {
    reply->set_version(server_version_);
  }
  return Status::OK;
}

auto
InferenceServiceImpl::ModelMetadata(
    ServerContext* /*context*/, const inference::ModelMetadataRequest* request,
    inference::ModelMetadataResponse* reply) -> Status
{
  const auto resolved_model_name = resolve_model_name(request->name());
  if (!resolved_model_name.empty()) {
    reply->set_name(resolved_model_name);
  }
  if (!request->version().empty()) {
    reply->add_versions(request->version());
  }

  if (const Status input_status = append_input_schema(
          expected_input_types_, expected_input_dims_, expected_input_names_,
          [&reply]() { return reply->add_inputs(); },
          [](auto& input, at::ScalarType type) -> Status {
            try {
              input.set_datatype(scalar_type_to_datatype(type));
              return Status::OK;
            }
            catch (const std::invalid_argument& exception) {
              return {grpc::StatusCode::INTERNAL, exception.what()};
            }
          },
          [](auto& input, int64_t dim) { input.add_shape(dim); });
      !input_status.ok()) {
    return input_status;
  }

  if (const Status output_status = append_output_schema(
          reference_outputs_, expected_output_names_,
          [&reply]() { return reply->add_outputs(); },
          [](auto& output, at::ScalarType type) -> Status {
            try {
              output.set_datatype(scalar_type_to_datatype(type));
              return Status::OK;
            }
            catch (const std::invalid_argument& exception) {
              return {grpc::StatusCode::INTERNAL, exception.what()};
            }
          },
          [](auto& output, int64_t dim) { output.add_shape(dim); });
      !output_status.ok()) {
    return output_status;
  }

  return Status::OK;
}

auto
InferenceServiceImpl::ModelConfig(
    ServerContext* /*context*/, const inference::ModelConfigRequest* request,
    inference::ModelConfigResponse* reply) -> Status
{
  auto* config = reply->mutable_config();
  const auto resolved_model_name = resolve_model_name(request->name());
  if (!resolved_model_name.empty()) {
    config->set_name(resolved_model_name);
  }
  config->set_max_batch_size(max_batch_size_);

  if (const Status input_status = append_input_schema(
          expected_input_types_, expected_input_dims_, expected_input_names_,
          [&config]() { return config->add_input(); },
          [](auto& input, at::ScalarType type) -> Status {
            const auto dtype = scalar_type_to_model_dtype(type);
            if (dtype == inference::DataType::TYPE_INVALID) {
              return {grpc::StatusCode::INTERNAL, "Unsupported input datatype"};
            }
            input.set_data_type(dtype);
            return Status::OK;
          },
          [](auto& input, int64_t dim) { input.add_dims(dim); });
      !input_status.ok()) {
    return input_status;
  }

  if (const Status output_status = append_output_schema(
          reference_outputs_, expected_output_names_,
          [&config]() { return config->add_output(); },
          [](auto& output, at::ScalarType type) -> Status {
            const auto dtype = scalar_type_to_model_dtype(type);
            if (dtype == inference::DataType::TYPE_INVALID) {
              return {
                  grpc::StatusCode::INTERNAL, "Unsupported output datatype"};
            }
            output.set_data_type(dtype);
            return Status::OK;
          },
          [](auto& output, int64_t dim) { output.add_dims(dim); });
      !output_status.ok()) {
    return output_status;
  }

  return Status::OK;
}

auto
InferenceServiceImpl::ModelStatistics(
    ServerContext* /*context*/,
    const inference::ModelStatisticsRequest* request,
    inference::ModelStatisticsResponse* reply) -> Status
{
  if (request == nullptr || reply == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "Invalid request"};
  }
  const std::string requested_name = resolve_model_name(request->name());
  const std::string requested_version = request->version();
  const auto fill_stat = [](inference::StatisticDuration* target,
                            const StatisticDurationState& state) {
    if (target == nullptr) {
      return;
    }
    target->set_count(state.count);
    target->set_ns(state.ns);
  };

  std::vector<std::pair<ModelStatsKey, ModelStatsState>> snapshot;
  {
    std::scoped_lock lock(model_stats_mutex_);
    snapshot.reserve(model_stats_.size());
    for (const auto& [key, state] : model_stats_) {
      snapshot.emplace_back(key, state);
    }
  }

  for (const auto& [key, state] : snapshot) {
    if (!requested_name.empty() && key.name != requested_name) {
      continue;
    }
    if (!requested_version.empty() && key.version != requested_version) {
      continue;
    }
    auto* stats = reply->add_model_stats();
    if (!key.name.empty()) {
      stats->set_name(key.name);
    }
    if (!key.version.empty()) {
      stats->set_version(key.version);
    }
    stats->set_last_inference(state.last_inference_ms);
    stats->set_inference_count(state.inference_count);
    stats->set_execution_count(state.execution_count);

    auto* infer_stats = stats->mutable_inference_stats();
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (testing::inference_service_test_internal::detail::
            model_statistics_force_null_target_flag_ref()) {
      fill_stat(nullptr, state.inference_stats.success);
    }
#endif  // SONAR_IGNORE_END
    fill_stat(infer_stats->mutable_success(), state.inference_stats.success);
    fill_stat(infer_stats->mutable_fail(), state.inference_stats.fail);
    fill_stat(infer_stats->mutable_queue(), state.inference_stats.queue);
    fill_stat(
        infer_stats->mutable_compute_input(),
        state.inference_stats.compute_input);
    fill_stat(
        infer_stats->mutable_compute_infer(),
        state.inference_stats.compute_infer);
    fill_stat(
        infer_stats->mutable_compute_output(),
        state.inference_stats.compute_output);
  }

  return Status::OK;
}

auto
InferenceServiceImpl::ModelStreamInfer(
    ServerContext* /*context*/,
    grpc::ServerReaderWriter<
        inference::ModelStreamInferResponse,
        inference::ModelInferRequest>* /*stream*/) -> Status
{
  return unimplemented_rpc_status("ModelStreamInfer");
}

auto
InferenceServiceImpl::RepositoryIndex(
    ServerContext* /*context*/,
    const inference::RepositoryIndexRequest* /*request*/,
    inference::RepositoryIndexResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("RepositoryIndex");
}

auto
InferenceServiceImpl::RepositoryModelLoad(
    ServerContext* /*context*/,
    const inference::RepositoryModelLoadRequest* /*request*/,
    inference::RepositoryModelLoadResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("RepositoryModelLoad");
}

auto
InferenceServiceImpl::RepositoryModelUnload(
    ServerContext* /*context*/,
    const inference::RepositoryModelUnloadRequest* /*request*/,
    inference::RepositoryModelUnloadResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("RepositoryModelUnload");
}

auto
InferenceServiceImpl::SystemSharedMemoryStatus(
    ServerContext* /*context*/,
    const inference::SystemSharedMemoryStatusRequest* /*request*/,
    inference::SystemSharedMemoryStatusResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("SystemSharedMemoryStatus");
}

auto
InferenceServiceImpl::SystemSharedMemoryRegister(
    ServerContext* /*context*/,
    const inference::SystemSharedMemoryRegisterRequest* /*request*/,
    inference::SystemSharedMemoryRegisterResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("SystemSharedMemoryRegister");
}

auto
InferenceServiceImpl::SystemSharedMemoryUnregister(
    ServerContext* /*context*/,
    const inference::SystemSharedMemoryUnregisterRequest* /*request*/,
    inference::SystemSharedMemoryUnregisterResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("SystemSharedMemoryUnregister");
}

auto
InferenceServiceImpl::CudaSharedMemoryStatus(
    ServerContext* /*context*/,
    const inference::CudaSharedMemoryStatusRequest* /*request*/,
    inference::CudaSharedMemoryStatusResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("CudaSharedMemoryStatus");
}

auto
InferenceServiceImpl::CudaSharedMemoryRegister(
    ServerContext* /*context*/,
    const inference::CudaSharedMemoryRegisterRequest* /*request*/,
    inference::CudaSharedMemoryRegisterResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("CudaSharedMemoryRegister");
}

auto
InferenceServiceImpl::CudaSharedMemoryUnregister(
    ServerContext* /*context*/,
    const inference::CudaSharedMemoryUnregisterRequest* /*request*/,
    inference::CudaSharedMemoryUnregisterResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("CudaSharedMemoryUnregister");
}

auto
InferenceServiceImpl::TraceSetting(
    ServerContext* /*context*/,
    const inference::TraceSettingRequest* /*request*/,
    inference::TraceSettingResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("TraceSetting");
}

auto
InferenceServiceImpl::LogSettings(
    ServerContext* /*context*/,
    const inference::LogSettingsRequest* /*request*/,
    inference::LogSettingsResponse* /*reply*/) -> Status
{
  return unimplemented_rpc_status("LogSettings");
}
