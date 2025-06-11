#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>
#include <string>

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

      std::cout << "First values of input tensor '" << input.name() << "': ";
      const int count = std::min(10, contents.fp32_contents_size());
      for (int i = 0; i < count; ++i) {
        std::cout << contents.fp32_contents(i) << ' ';
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