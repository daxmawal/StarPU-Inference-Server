#include "inference_service.hpp"

#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <string>

#include "utils/client_utils.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using inference::ModelInferRequest;
using inference::ModelInferResponse;
using inference::ServerLiveRequest;
using inference::ServerLiveResponse;

std::unique_ptr<Server> g_server;

namespace {
// Convert gRPC input to torch::Tensor
auto
convert_input_to_tensor(
    const ModelInferRequest::InferInputTensor& input, const std::string& raw)
{
  std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
  auto tensor = torch::empty(shape, torch::kFloat32);
  float* dest_ptr = tensor.data_ptr<float>();
  std::memcpy(dest_ptr, raw.data(), raw.size());

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
    out_tensor->set_name("output" + std::to_string(idx));
    out_tensor->set_datatype("FP32");
    for (const auto dim : out.sizes()) {
      out_tensor->add_shape(dim);
    }

    auto flat = out.view({-1});
    reply->add_raw_output_contents()->assign(
        reinterpret_cast<const char*>(flat.data_ptr<float>()),
        flat.numel() * sizeof(float));
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
InferenceServiceImpl::ModelInfer(
    ServerContext* /*context*/, const ModelInferRequest* request,
    ModelInferResponse* reply) -> Status
{
  auto recv_tp = std::chrono::high_resolution_clock::now();
  auto recv_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     recv_tp.time_since_epoch())
                     .count();

  if (request->raw_input_contents_size() != request->inputs_size()) {
    return Status(
        grpc::StatusCode::INVALID_ARGUMENT,
        "Number of raw inputs does not match number of input tensors");
  }

  std::vector<torch::Tensor> inputs;
  inputs.reserve(request->inputs_size());
  for (int i = 0; i < request->inputs_size(); ++i) {
    const auto& input = request->inputs(i);
    const auto& raw = request->raw_input_contents(i);

    size_t expected = sizeof(float);
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

  auto job =
      client_utils::create_job(inputs, *reference_outputs_, next_job_id_++);
  std::promise<std::vector<torch::Tensor>> result_promise;
  auto result_future = result_promise.get_future();

  job->set_on_complete(
      [&result_promise](const std::vector<torch::Tensor>& outs, double) {
        result_promise.set_value(outs);
      });

  queue_->push(job);
  auto outputs = result_future.get();

  reply->set_model_name(request->model_name());
  reply->set_model_version(request->model_version());
  fill_output_tensor(reply, outputs);

  auto send_tp = std::chrono::high_resolution_clock::now();
  auto send_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     send_tp.time_since_epoch())
                     .count();

  reply->set_server_receive_ms(recv_ms);
  reply->set_server_send_ms(send_ms);

  return Status::OK;
}

void
RunServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs)
{
  const std::string server_address("0.0.0.0:50051");
  InferenceServiceImpl service(&queue, &reference_outputs);

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  const int max_msg_size = 32 * 1024 * 1024;
  builder.SetMaxReceiveMessageSize(max_msg_size);
  builder.SetMaxSendMessageSize(max_msg_size);

  g_server = builder.BuildAndStart();
  std::cout << "Server listening on " << server_address << std::endl;
  g_server->Wait();
}
