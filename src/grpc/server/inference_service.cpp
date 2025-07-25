#include "inference_service.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <string>
#include <unordered_map>

#include "utils/client_utils.hpp"
#include "utils/datatype_utils.hpp"

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

// Convert gRPC input to torch::Tensor
auto
convert_input_to_tensor(
    const ModelInferRequest::InferInputTensor& input, const std::string& raw)
{
  std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
  const auto dtype = datatype_to_scalar_type(input.datatype());
  auto tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype));
  std::memcpy(tensor.data_ptr(), raw.data(), raw.size());

  return tensor;
}

// Fill gRPC output from torch::Tensor
void
fill_output_tensor(
    ModelInferResponse* reply, const std::vector<torch::Tensor>& outputs)
{
  for (size_t idx = 0; idx < outputs.size(); ++idx) {
    const auto& out = outputs[idx].to(torch::kCPU);
    auto* out_tensor = reply->add_outputs();
    out_tensor->set_name(std::format("output{}", idx));
    out_tensor->set_datatype(scalar_type_to_datatype(out.scalar_type()));
    for (const auto dim : out.sizes()) {
      out_tensor->add_shape(dim);
    }

    auto flat = out.view({-1});
    reply->add_raw_output_contents()->assign(
        reinterpret_cast<const char*>(flat.data_ptr()),
        flat.numel() * flat.element_size());
  }
}
}  // namespace

// Implementation
InferenceServiceImpl::InferenceServiceImpl(
    InferenceQueue* queue, const std::vector<torch::Tensor>* reference_outputs)
    : queue_(queue), reference_outputs_(reference_outputs)
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
  if (request->raw_input_contents_size() != request->inputs_size()) {
    return Status(
        grpc::StatusCode::INVALID_ARGUMENT,
        "Number of raw inputs does not match number of input tensors");
  }

  inputs.reserve(request->inputs_size());
  for (int i = 0; i < request->inputs_size(); ++i) {
    const auto& input = request->inputs(i);
    const auto& raw = request->raw_input_contents(i);

    const auto dtype = datatype_to_scalar_type(input.datatype());
    size_t expected = element_size(dtype);
    for (const auto dim : input.shape()) {
      expected *= static_cast<size_t>(dim);
    }
    if (expected != raw.size()) {
      return Status(
          grpc::StatusCode::INVALID_ARGUMENT,
          "Input tensor shape does not match raw content size");
    }

    inputs.push_back(convert_input_to_tensor(input, raw));
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

  queue_->push(job);
  outputs = result_future.get();

  if (outputs.empty()) {
    return Status(grpc::StatusCode::INTERNAL, "Inference failed");
  }

  return Status::OK;
}

void
InferenceServiceImpl::populate_response(
    const ModelInferRequest* request, ModelInferResponse* reply,
    const std::vector<torch::Tensor>& outputs, int64_t recv_ms, int64_t send_ms)
{
  reply->set_model_name(request->model_name());
  reply->set_model_version(request->model_version());
  reply->set_server_receive_ms(recv_ms);
  reply->set_server_send_ms(send_ms);
  fill_output_tensor(reply, outputs);
}

auto
InferenceServiceImpl::ModelInfer(
    ServerContext* /*context*/, const ModelInferRequest* request,
    ModelInferResponse* reply) -> Status
{
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

  populate_response(request, reply, outputs, recv_ms, send_ms);
  return Status::OK;
}

void
RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::string& address, int max_message_bytes,
    std::unique_ptr<Server>& server)
{
  InferenceServiceImpl service(&queue, &reference_outputs);

  ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  builder.SetMaxReceiveMessageSize(max_message_bytes);
  builder.SetMaxSendMessageSize(max_message_bytes);

  server = builder.BuildAndStart();
  std::cout << "Server listening on " << address << std::endl;
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
