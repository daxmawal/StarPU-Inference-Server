#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>

#include "grpc_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using inference::GRPCInferenceService;
using inference::ModelInferRequest;
using inference::ModelInferResponse;
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

  bool ModelInfer()
  {
    ModelInferRequest request;
    request.set_model_name("example");
    request.set_model_version("1");

    auto* input = request.add_inputs();
    input->set_name("input");
    input->set_datatype("FP32");
    input->add_shape(16);

    auto* contents = input->mutable_contents();
    for (int i = 0; i < 16; ++i) {
      contents->add_fp32_contents(static_cast<float>(i));
    }

    ModelInferResponse response;
    ClientContext context;

    Status status = stub_->ModelInfer(&context, request, &response);

    if (status.ok()) {
      std::cout << "ModelInfer call succeeded" << std::endl;
      return true;
    } else {
      std::cerr << "ModelInfer RPC failed: " << status.error_message()
                << std::endl;
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
  client.ModelInfer();
  return 0;
}