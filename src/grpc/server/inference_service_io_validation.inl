using TensorDataByte = std::byte;
using TensorDataPtr = TensorDataByte*;

auto
parse_input_dtype(
    const inference::ModelInferRequest::InferInputTensor& input,
    at::ScalarType expected, at::ScalarType& out) -> Status
{
  try {
    out = datatype_to_scalar_type(input.datatype());
  }
  catch (const std::invalid_argument& e) {
    return {grpc::StatusCode::INVALID_ARGUMENT, e.what()};
  }

  if (out != expected) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT, "Input tensor datatype mismatch"};
  }

  return Status::OK;
}

auto
validate_configured_shape(
    const std::vector<int64_t>& shape, const std::vector<int64_t>& expected,
    bool batching_allowed, int max_batch_size) -> Status
{
  const auto rank = static_cast<int64_t>(shape.size());
  const auto expected_rank = static_cast<int64_t>(expected.size());

  auto tails_match = [&](int64_t shape_offset, int64_t expected_offset) {
    if (rank - shape_offset != expected_rank - expected_offset) {
      return false;
    }
    for (int64_t idx = 0; idx < rank - shape_offset; ++idx) {
      if (shape[static_cast<size_t>(shape_offset + idx)] !=
          expected[static_cast<size_t>(expected_offset + idx)]) {
        return false;
      }
    }
    return true;
  };

  if (!batching_allowed) {
    if (tails_match(0, 0)) {
      return Status::OK;
    }
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "Input tensor shape does not match configured dimensions or batch "
        "limits"};
  }

  if (rank == 0) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "Input tensor shape does not match configured dimensions or batch "
        "limits"};
  }

  if (tails_match(0, 0)) {
    return Status::OK;
  }

  auto validate_batch_size = [&](int64_t batch_size) -> Status {
    if (batch_size <= 0) {
      return {
          grpc::StatusCode::INVALID_ARGUMENT,
          "Input tensor shape does not match configured dimensions or batch "
          "limits (batch size must be positive)"};
    }
    if (batch_size > max_batch_size) {
      return {
          grpc::StatusCode::INVALID_ARGUMENT,
          std::format(
              "Input tensor shape does not match configured dimensions or "
              "batch limits (batch size {} exceeds configured max of {})",
              batch_size, max_batch_size)};
    }
    return Status::OK;
  };

  if (rank >= 1 && tails_match(1, 1)) {
    const int64_t batch_size = shape.front();
    if (auto status = validate_batch_size(batch_size); !status.ok()) {
      return status;
    }
    return Status::OK;
  }

  if (rank >= 1 && tails_match(1, 0)) {
    const int64_t batch_size = shape.front();
    if (auto status = validate_batch_size(batch_size); !status.ok()) {
      return status;
    }
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "Input tensor shape does not match configured dimensions or batch "
        "limits"};
  }

  return {
      grpc::StatusCode::INVALID_ARGUMENT,
      "Input tensor shape does not match configured dimensions or batch "
      "limits"};
}

auto
checked_mul(size_t lhs, size_t rhs) -> std::optional<size_t>
{
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    return std::nullopt;
  }
  return lhs * rhs;
}

auto
resolve_output_names(
    std::span<const std::string> output_names,
    std::size_t output_count) -> std::vector<std::string>
{
  std::vector<std::string> resolved;
  resolved.reserve(output_count);
  for (std::size_t i = 0; i < output_count; ++i) {
    if (i < output_names.size() && !output_names[i].empty()) {
      resolved.push_back(output_names[i]);
    } else {
      resolved.push_back(std::format("output{}", i));
    }
  }
  return resolved;
}

auto
convert_input_to_tensor(
    const ModelInferRequest::InferInputTensor& input,
    const std::vector<int64_t>& shape, const std::string& raw,
    at::ScalarType dtype, torch::Tensor& tensor,
    std::shared_ptr<const void>* keep_alive) -> Status
{
  auto options = torch::TensorOptions().dtype(dtype);

  std::optional<size_t> expected = element_size(dtype);
  for (const auto dim : input.shape()) {
    if (dim <= 0) {
      return {
          grpc::StatusCode::INVALID_ARGUMENT,
          "Input tensor shape contains non-positive dimension"};
    }
    expected = checked_mul(*expected, static_cast<size_t>(dim));
    if (!expected) {
      return {
          grpc::StatusCode::INVALID_ARGUMENT,
          "Input tensor shape is too large"};
    }
  }
  if (*expected != raw.size()) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "Raw input size does not match tensor size"};
  }

  auto buffer = std::make_shared<std::vector<TensorDataByte>>(raw.size());
  if (!raw.empty()) {
    std::memcpy(buffer->data(), raw.data(), raw.size());
  }
  auto deleter = [buffer](auto* /*unused*/) mutable { buffer.reset(); };

  TensorDataPtr tensor_data = buffer->data();
  tensor = torch::from_blob(tensor_data, shape, deleter, options);
  if (keep_alive != nullptr) {
    *keep_alive = std::shared_ptr<const void>(buffer, buffer->data());
  }
  return Status::OK;
}

struct InputNameState {
  bool any_named = false;
  bool any_unnamed = false;
};

auto
validate_input_counts(
    const ModelInferRequest* request, std::size_t expected_count) -> Status
{
  if (request == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "ModelInfer request is null"};
  }
  if (request->inputs_size() != static_cast<int>(expected_count)) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        std::format(
            "Expected {} input tensors but received {}", expected_count,
            request->inputs_size())};
  }
  if (request->raw_input_contents_size() != request->inputs_size()) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "Number of raw inputs does not match number of input tensors"};
  }
  return Status::OK;
}

auto
collect_input_name_state(const ModelInferRequest* request) -> InputNameState
{
  InputNameState state{};
  for (const auto& input : request->inputs()) {
    if (input.name().empty()) {
      state.any_unnamed = true;
    } else {
      state.any_named = true;
    }
  }
  return state;
}

auto
validate_input_names(const InputNameState& state, bool has_expected_names)
    -> Status
{
  if (state.any_named && state.any_unnamed) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "All input tensors must include a name when using named inputs"};
  }
  if (has_expected_names && !state.any_named) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "Input tensor names must be provided"};
  }
  return Status::OK;
}

auto
build_expected_index_by_name(
    const std::vector<std::string>& expected_input_names,
    std::unordered_map<std::string_view, std::size_t>& expected_index_by_name)
    -> Status
{
  expected_index_by_name.reserve(expected_input_names.size());
  for (std::size_t i = 0; i < expected_input_names.size(); ++i) {
    const auto& name = expected_input_names[i];
    if (name.empty()) {
      return {
          grpc::StatusCode::INVALID_ARGUMENT,
          std::format("Configured input name missing at index {}", i)};
    }
    const auto [existing_it, inserted] =
        expected_index_by_name.try_emplace(name, i);
    if (!inserted) {
      return {
          grpc::StatusCode::INVALID_ARGUMENT,
          std::format(
              "Configured input name '{}' is duplicated", existing_it->first)};
    }
  }
  return Status::OK;
}

auto
resolve_expected_index(
    const ModelInferRequest::InferInputTensor& input, int input_index,
    const std::unordered_map<std::string_view, std::size_t>*
        expected_index_by_name,
    std::vector<bool>& filled, std::size_t& expected_index) -> Status
{
  expected_index = static_cast<std::size_t>(input_index);
  if (expected_index_by_name == nullptr) {
    return Status::OK;
  }
  const auto name_iter = expected_index_by_name->find(input.name());
  if (name_iter == expected_index_by_name->end()) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        std::format("Unexpected input tensor name '{}'", input.name())};
  }
  expected_index = name_iter->second;
  if (filled[expected_index]) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        std::format("Input tensor name '{}' is duplicated", input.name())};
  }
  return Status::OK;
}

struct ProcessInputContext {
  const std::unordered_map<std::string_view, std::size_t>*
      expected_index_by_name;
  const std::vector<at::ScalarType>* expected_input_types;
  const std::vector<std::vector<int64_t>>* expected_input_dims;
  int max_batch_size;
  std::vector<torch::Tensor>* ordered_inputs;
  std::vector<std::shared_ptr<const void>>* ordered_lifetimes;
  std::vector<bool>* filled;
};

auto
process_input(
    const ModelInferRequest* request, int input_index,
    const ProcessInputContext& context) -> Status
{
  if (request == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "ModelInfer request is null"};
  }
  const auto& input = request->inputs(input_index);
  const auto& raw = request->raw_input_contents(input_index);
  std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

  std::size_t expected_index = 0;
  Status status = resolve_expected_index(
      input, input_index, context.expected_index_by_name, *context.filled,
      expected_index);
  if (!status.ok()) {
    return status;
  }

  at::ScalarType dtype = at::kFloat;
  status = parse_input_dtype(
      input, (*context.expected_input_types)[expected_index], dtype);
  if (!status.ok()) {
    return status;
  }

  torch::Tensor tensor;
  std::shared_ptr<const void> tensor_guard;
  status = convert_input_to_tensor(
      input, shape, raw, dtype, tensor,
      context.ordered_lifetimes != nullptr ? &tensor_guard : nullptr);
  if (!status.ok()) {
    return status;
  }

  if (expected_index < context.expected_input_dims->size()) {
    const auto& expected_dims = (*context.expected_input_dims)[expected_index];
    const bool batching_allowed = (context.max_batch_size > 0);
    status = validate_configured_shape(
        shape, expected_dims, batching_allowed, context.max_batch_size);
    if (!status.ok()) {
      return status;
    }
  }

  (*context.ordered_inputs)[expected_index] = std::move(tensor);
  if (context.ordered_lifetimes != nullptr) {
    (*context.ordered_lifetimes)[expected_index] = std::move(tensor_guard);
  }
  (*context.filled)[expected_index] = true;
  return Status::OK;
}

auto
fill_output_tensor(
    ModelInferResponse* reply, const std::vector<torch::Tensor>& outputs,
    std::span<const std::size_t> output_indices,
    std::span<const std::string> output_names) -> Status
{
  if (reply == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "ModelInfer response is null"};
  }
  for (const std::size_t output_index : output_indices) {
    if (output_index >= outputs.size()) {
      return {
          grpc::StatusCode::INVALID_ARGUMENT,
          "Requested output index out of range"};
    }
    const auto& original = outputs[output_index];
    torch::Tensor out = original;
    if (!original.device().is_cpu()) {
      out = original.to(torch::kCPU);
    }
    auto* out_tensor = reply->add_outputs();
    if (output_index < output_names.size()) {
      out_tensor->set_name(output_names[output_index]);
    } else {
      out_tensor->set_name(std::format("output{}", output_index));
    }
    out_tensor->set_datatype(scalar_type_to_datatype(out.scalar_type()));
    for (const auto dim : out.sizes()) {
      out_tensor->add_shape(dim);
    }

    auto flat = out.contiguous().view({-1});
    const auto total_bytes =
        checked_mul(static_cast<size_t>(flat.numel()), flat.element_size());
    if (!total_bytes.has_value()) {
      return {
          grpc::StatusCode::INVALID_ARGUMENT, "Output tensor size overflow"};
    }
    reply->add_raw_output_contents()->assign(
        static_cast<const char*>(flat.data_ptr()), *total_bytes);
  }
  return Status::OK;
}
