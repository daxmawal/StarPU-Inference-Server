#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>

#include "grpc_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using inference::GRPCInferenceService;
using inference::ServerLiveRequest;
using inference::ServerLiveResponse;

class InferenceClient {
 public:
  InferenceClient(std::shared_ptr<Channel> channel)
      : stub_(GRPCInferenceService::NewStub(channel))
  {
  }

  bool ServerIsLive()
  {
    ServerLiveRequest request;
    ServerLiveResponse response;
    ClientContext context;

    Status status = stub_->ServerLive(&context, request, &response);

    if (status.ok()) {
      std::cout << "Server live: " << std::boolalpha << response.live()
                << std::endl;
      return response.live();
    } else {
      std::cerr << "RPC failed: " << status.error_message() << std::endl;
      return false;
    }
  }

 private:
  std::unique_ptr<GRPCInferenceService::Stub> stub_;
};

int
main()
{
  InferenceClient client(grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials()));
  client.ServerIsLive();
  return 0;
}
