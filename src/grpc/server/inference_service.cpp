#include "inference_service.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <format>
#include <future>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>

#include "monitoring/metrics.hpp"
#include "utils/client_utils.hpp"
#include "utils/datatype_utils.hpp"
#include "utils/logger.hpp"

namespace starpu_server {
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using inference::ModelInferRequest;
using inference::ModelInferResponse;
using inference::ModelReadyRequest;
using inference::ModelReadyResponse;
using inference::ServerLiveRequest;
using inference::ServerLiveResponse;
using inference::ServerReadyRequest;
using inference::ServerReadyResponse;


namespace {

auto
checked_mul(size_t lhs, size_t rhs) -> std::optional<size_t>
{
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    return std::nullopt;
  }
  return lhs * rhs;
}

auto
convert_input_to_tensor(
    const ModelInferRequest::InferInputTensor& input, const std::string& raw,
    at::ScalarType dtype, torch::Tensor& tensor) -> Status
{
  std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
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

  tensor = torch::empty(shape, options);
  std::memcpy(tensor.data_ptr(), raw.data(), raw.size());
  return Status::OK;
}

auto
fill_output_tensor(
    ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs) -> Status
{
  for (size_t idx = 0; idx < outputs.size(); ++idx) {
    const auto& out = outputs[idx].to(torch::kCPU);
    auto* out_tensor = reply->add_outputs();
    out_tensor->set_name(std::format("output{}", idx));
    out_tensor->set_datatype(scalar_type_to_datatype(out.scalar_type()));
    for (const auto dim : out.sizes()) {
      out_tensor->add_shape(dim);
    }

    auto flat = out.contiguous().view({-1});
    const auto total_bytes =
        checked_mul(static_cast<size_t>(flat.numel()), flat.element_size());
    if (!total_bytes) {
      return {
          grpc::StatusCode::INVALID_ARGUMENT, "Output tensor size overflow"};
    }
    reply->add_raw_output_contents()->assign(
        static_cast<const char*>(flat.data_ptr()), *total_bytes);
  }
  return Status::OK;
}
}  // namespace

InferenceServiceImpl::InferenceServiceImpl(
    InferenceQueue* queue, const std::vector<torch::Tensor>* reference_outputs,
    std::vector<at::ScalarType> expected_input_types,
    std::vector<std::vector<int64_t>> expected_input_dims, int max_batch_size)
    : queue_(queue), reference_outputs_(reference_outputs),
      expected_input_types_(std::move(expected_input_types)),
      expected_input_dims_(std::move(expected_input_dims)),
      max_batch_size_(max_batch_size)
{
}

InferenceServiceImpl::InferenceServiceImpl(
    InferenceQueue* queue, const std::vector<torch::Tensor>* reference_outputs,
    std::vector<at::ScalarType> expected_input_types)
    : queue_(queue), reference_outputs_(reference_outputs),
      expected_input_types_(std::move(expected_input_types))
{
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
  reply->set_ready(true);
  return Status::OK;
}

auto
InferenceServiceImpl::ModelReady(
    ServerContext* /*context*/, const ModelReadyRequest* /*request*/,
    ModelReadyResponse* reply) -> Status
{
  reply->set_ready(true);
  return Status::OK;
}

auto
InferenceServiceImpl::validate_and_convert_inputs(
    const ModelInferRequest* request,
    std::vector<torch::Tensor>& inputs) -> Status
{
  if (request->inputs_size() !=
      static_cast<int>(expected_input_types_.size())) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        std::format(
            "Expected {} input tensors but received {}",
            expected_input_types_.size(), request->inputs_size())};
  }
  if (request->raw_input_contents_size() != request->inputs_size()) {
    return {
        grpc::StatusCode::INVALID_ARGUMENT,
        "Number of raw inputs does not match number of input tensors"};
  }

  inputs.reserve(request->inputs_size());
  for (int i = 0; i < request->inputs_size(); ++i) {
    const auto& input = request->inputs(i);
    const auto& raw = request->raw_input_contents(i);

    at::ScalarType dtype = at::kFloat;
    try {
      dtype = datatype_to_scalar_type(input.datatype());
    }
    catch (const std::invalid_argument& e) {
      return {grpc::StatusCode::INVALID_ARGUMENT, e.what()};
    }

    if (dtype != expected_input_types_[i]) {
      return {
          grpc::StatusCode::INVALID_ARGUMENT, "Input tensor datatype mismatch"};
    }

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
          "Input tensor shape does not match raw content size"};
    }

    if (static_cast<size_t>(i) < expected_input_dims_.size()) {
      const std::vector<int64_t>& exp =
          expected_input_dims_[static_cast<size_t>(i)];
      std::vector<int64_t> shp(input.shape().begin(), input.shape().end());

      const auto rank = static_cast<int64_t>(shp.size());
      const auto exp_rank = static_cast<int64_t>(exp.size());
      const bool batching_allowed = (max_batch_size_ > 0);

      auto check_tail_eq = [&](int64_t offset_in, int64_t offset_exp) -> bool {
        if (rank - offset_in != exp_rank - offset_exp)
          return false;
        for (int64_t k = 0; k < rank - offset_in; ++k) {
          if (shp[static_cast<size_t>(offset_in + k)] !=
              exp[static_cast<size_t>(offset_exp + k)]) {
            return false;
          }
        }
        return true;
      };

      bool shape_ok = false;
      if (!batching_allowed) {
        shape_ok = (rank == exp_rank) && check_tail_eq(0, 0);
      } else {
        if (rank == exp_rank) {
          if (rank >= 1 && check_tail_eq(1, 1)) {
            const int64_t b = shp[0];
            shape_ok = (b >= 1 && b <= max_batch_size_);
          }
        } else if (rank == exp_rank + 1) {
          if (check_tail_eq(1, 0)) {
            const int64_t b = shp[0];
            shape_ok = (b >= 1 && b <= max_batch_size_);
          }
        }
      }

      if (!shape_ok) {
        return {
            grpc::StatusCode::INVALID_ARGUMENT,
            "Input tensor shape does not match configured dimensions or batch "
            "limits"};
      }
    }

    torch::Tensor tensor;
    Status status = convert_input_to_tensor(input, raw, dtype, tensor);
    if (!status.ok()) {
      return status;
    }
    inputs.push_back(std::move(tensor));
  }
  return Status::OK;
}

auto
InferenceServiceImpl::submit_job_and_wait(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) -> Status
{
  auto job =
      client_utils::create_job(inputs, *reference_outputs_, next_job_id_++);
  std::promise<std::vector<torch::Tensor>> result_promise;
  auto result_future = result_promise.get_future();

  job->set_on_complete(
      [&result_promise](const std::vector<torch::Tensor>& outs, double) {
        result_promise.set_value(outs);
      });

  const bool pushed = queue_->push(job);
  if (!pushed) {
    return {grpc::StatusCode::UNAVAILABLE, "Inference queue shutting down"};
  }

  const auto timeout = std::chrono::seconds(30);
  if (result_future.wait_for(timeout) != std::future_status::ready) {
    return {
        grpc::StatusCode::DEADLINE_EXCEEDED,
        "Inference result not available in time"};
  }
  outputs = result_future.get();

  if (outputs.empty()) {
    return {grpc::StatusCode::INTERNAL, "Inference failed"};
  }

  return Status::OK;
}

auto
InferenceServiceImpl::populate_response(
    const ModelInferRequest* request, ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs, int64_t recv_ms,
    int64_t send_ms) -> Status
{
  reply->set_model_name(request->model_name());
  reply->set_model_version(request->model_version());
  reply->set_server_receive_ms(recv_ms);
  reply->set_server_send_ms(send_ms);
  return fill_output_tensor(reply, outputs);
}

auto
InferenceServiceImpl::ModelInfer(
    ServerContext* /*context*/, const ModelInferRequest* request,
    ModelInferResponse* reply) -> Status
{
  auto metrics = get_metrics();
  if (metrics && metrics->requests_total != nullptr) {
    metrics->requests_total->Increment();
  }

  auto recv_tp = std::chrono::high_resolution_clock::now();
  int64_t recv_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        recv_tp.time_since_epoch())
                        .count();

  std::vector<torch::Tensor> inputs;
  Status status = validate_and_convert_inputs(request, inputs);
  if (!status.ok()) {
    return status;
  }

  std::vector<torch::Tensor> outputs;
  status = submit_job_and_wait(inputs, outputs);
  if (!status.ok()) {
    return status;
  }

  auto send_tp = std::chrono::high_resolution_clock::now();
  int64_t send_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        send_tp.time_since_epoch())
                        .count();

  status = populate_response(request, reply, outputs, recv_ms, send_ms);
  if (!status.ok()) {
    return status;
  }

  if (metrics && metrics->inference_latency != nullptr) {
    const auto latency_ms =
        std::chrono::duration<double, std::milli>(send_tp - recv_tp).count();
    metrics->inference_latency->Observe(latency_ms);
  }
  return Status::OK;
}

void
RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::vector<at::ScalarType>& expected_input_types,
    const std::vector<std::vector<int64_t>>& expected_input_dims,
    int max_batch_size, const std::string& address,
    std::size_t max_message_bytes, VerbosityLevel verbosity,
    std::unique_ptr<Server>& server)
{
  InferenceServiceImpl service(
      &queue, &reference_outputs, expected_input_types, expected_input_dims,
      max_batch_size);

  ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  const int grpc_max_message_bytes =
      max_message_bytes >
              static_cast<std::size_t>(std::numeric_limits<int>::max())
          ? std::numeric_limits<int>::max()
          : static_cast<int>(max_message_bytes);
  builder.SetMaxReceiveMessageSize(grpc_max_message_bytes);
  builder.SetMaxSendMessageSize(grpc_max_message_bytes);

  server = builder.BuildAndStart();
  if (!server) {
    log_error(std::format("Failed to start gRPC server on {}", address));
    return;
  }
  log_info(verbosity, std::format("Server listening on {}", address));
  server->Wait();
  server.reset();
}

void
RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::vector<at::ScalarType>& expected_input_types,
    const std::string& address, std::size_t max_message_bytes,
    VerbosityLevel verbosity, std::unique_ptr<Server>& server)
{
  InferenceServiceImpl service(
      &queue, &reference_outputs,
      std::vector<at::ScalarType>(
          expected_input_types.begin(), expected_input_types.end()));

  ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  const int grpc_max_message_bytes =
      max_message_bytes >
              static_cast<std::size_t>(std::numeric_limits<int>::max())
          ? std::numeric_limits<int>::max()
          : static_cast<int>(max_message_bytes);
  builder.SetMaxReceiveMessageSize(grpc_max_message_bytes);
  builder.SetMaxSendMessageSize(grpc_max_message_bytes);

  server = builder.BuildAndStart();
  if (!server) {
    log_error(std::format("Failed to start gRPC server on {}", address));
    return;
  }
  log_info(verbosity, std::format("Server listening on {}", address));
  server->Wait();
  server.reset();
}

void
StopServer(std::unique_ptr<Server>& server)
{
  if (server) {
    server->Shutdown();
  }
}
}  // namespace starpu_server
