#include <grpcpp/grpcpp.h>

#include "inference.grpc.pb.h"
#include "inference.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using inference::InferenceRequest;
using inference::InferenceResponse;
using inference::InferenceService;

class InferenceServiceImpl final : public InferenceService::Service {
  Status RunInference(
      ServerContext* context, const InferenceRequest* request,
      InferenceResponse* response) override
  {
    std::string input = request->tensor_data();
    std::string output = input;  // simulate inference: identity

    response->set_tensor_data(output);
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

  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}


auto
main(int argc, char* argv[]) -> int
{
  RunServer();
  return 0;
}
