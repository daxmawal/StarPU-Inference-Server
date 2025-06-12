#include <ATen/ATen.h>
#include <grpcpp/grpcpp.h>
#include <torch/script.h>

#include <iostream>
#include <memory>
#include <string>

#include "core/starpu_setup.hpp"
#include "grpc_service.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using inference::GRPCInferenceService;
using inference::ModelInferRequest;
using inference::ModelInferResponse;
using inference::ServerLiveRequest;
using inference::ServerLiveResponse;

class InferenceServiceImpl final : public GRPCInferenceService::Service {
 public:
  Status ServerLive(
      ServerContext* context, const ServerLiveRequest* request,
      ServerLiveResponse* reply) override
  {
    std::cout << "Received ServerLive request" << std::endl;
    reply->set_live(true);
    return Status::OK;
  }

  Status ModelInfer(
      ServerContext* context, const ModelInferRequest* request,
      ModelInferResponse* reply) override
  {
    std::cout << "Received ModelInfer request" << std::endl;

    if (request->inputs_size() > 0) {
      const auto& input = request->inputs(0);
      const auto& contents = input.contents();

      std::vector<int64_t> shape;
      shape.reserve(input.shape_size());
      for (int i = 0; i < input.shape_size(); ++i) {
        shape.push_back(input.shape(i));
      }

      std::vector<float> data;
      data.reserve(contents.fp32_contents_size());
      for (int i = 0; i < contents.fp32_contents_size(); ++i) {
        data.push_back(contents.fp32_contents(i));
      }

      auto tensor = torch::from_blob(data.data(), shape, torch::kFloat).clone();

      std::cout << "Received tensor of size " << tensor.sizes() << std::endl;
      std::cout << "First values of input tensor '" << input.name() << "': ";
      const int count = std::min<int64_t>(10, tensor.numel());
      for (int i = 0; i < count; ++i) {
        std::cout << tensor.view({-1})[i].item<float>() << ' ';
      }
      std::cout << std::endl;
    }

    reply->set_model_name(request->model_name());
    reply->set_model_version(request->model_version());
    return Status::OK;
  }
};

void
RunServer()
{
  std::string server_address("0.0.0.0:50051");
  InferenceServiceImpl service;

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  const int max_msg_size = 32 * 1024 * 1024;  // 32MB, adjust as needed
  builder.SetMaxReceiveMessageSize(max_msg_size);
  builder.SetMaxSendMessageSize(max_msg_size);

  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listen on " << server_address << std::endl;
  server->Wait();
}

int
main()
{
  RunServer();
  return 0;
}