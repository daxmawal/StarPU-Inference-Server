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
using inference::ServerLiveRequest;
using inference::ServerLiveResponse;

class InferenceServiceImpl final : public GRPCInferenceService::Service {
 public:
  Status ServerLive(
      ServerContext* context, const ServerLiveRequest* request,
      ServerLiveResponse* reply) override
  {
    std::cout << "Received ServerLive request" << std::endl;
    reply->set_live(true);  // On simule que le serveur est "vivant"
    return Status::OK;
  }

  // Tu peux aussi implémenter ServerReady, ModelInfer, etc.
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
  std::cout << "Serveur d'inférence en écoute sur " << server_address
            << std::endl;
  server->Wait();
}

int
main()
{
  RunServer();
  return 0;
}
