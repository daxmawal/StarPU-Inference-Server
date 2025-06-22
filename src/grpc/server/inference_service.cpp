#include "inference_service.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <string>
#include <unordered_map>

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
// Map datatype strings to torch::ScalarType
auto
datatype_to_scalar_type(const std::string& dtype) -> at::ScalarType
{
  static const std::unordered_map<std::string, at::ScalarType> type_map = {
      {"FP32", at::kFloat},    {"FP64", at::kDouble}, {"FP16", at::kHalf},
      {"BF16", at::kBFloat16}, {"INT32", at::kInt},   {"INT64", at::kLong},
      {"INT16", at::kShort},   {"INT8", at::kChar},   {"UINT8", at::kByte},
      {"BOOL", at::kBool}};

  const auto it = type_map.find(dtype);
  if (it == type_map.end()) {
    throw std::invalid_argument("Unsupported tensor datatype: " + dtype);
  }
  return it->second;
}

auto
scalar_type_to_datatype(at::ScalarType type) -> std::string
{
  switch (type) {
    case at::kFloat:
      return "FP32";
    case at::kDouble:
      return "FP64";
    case at::kHalf:
      return "FP16";
    case at::kBFloat16:
      return "BF16";
    case at::kInt:
      return "INT32";
    case at::kLong:
      return "INT64";
    case at::kShort:
      return "INT16";
    case at::kChar:
      return "INT8";
    case at::kByte:
      return "UINT8";
    case at::kBool:
      return "BOOL";
    default:
      return "FP32";
  }
}

auto
element_size(at::ScalarType type) -> size_t
{
  switch (type) {
    case at::kFloat:
      return sizeof(float);
    case at::kDouble:
      return sizeof(double);
    case at::kHalf:
    case at::kBFloat16:
      return 2u;
    case at::kInt:
      return sizeof(int32_t);
    case at::kLong:
      return sizeof(int64_t);
    case at::kShort:
      return sizeof(int16_t);
    case at::kChar:
      return sizeof(int8_t);
    case at::kByte:
      return sizeof(uint8_t);
    case at::kBool:
      return sizeof(bool);
    default:
      return sizeof(float);
  }
}

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
    out_tensor->set_name("output" + std::to_string(idx));
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
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::string& address, int max_message_bytes)
{
  InferenceServiceImpl service(&queue, &reference_outputs);

  ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  builder.SetMaxReceiveMessageSize(max_message_bytes);
  builder.SetMaxSendMessageSize(max_message_bytes);

  g_server = builder.BuildAndStart();
  std::cout << "Server listening on " << address << std::endl;
  g_server->Wait();
}
