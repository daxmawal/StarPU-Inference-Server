#include <grpcpp/grpcpp.h>
#include <torch/script.h>

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
    input->add_shape(32);
    input->add_shape(3);
    input->add_shape(224);
    input->add_shape(224);

    auto tensor = torch::rand({32, 3, 224, 224}, torch::kFloat32);
    auto flat = tensor.view({-1});

    auto* contents = input->mutable_contents();
    contents->mutable_fp32_contents()->Reserve(flat.numel());
    for (int64_t i = 0; i < flat.numel(); ++i) {
      contents->add_fp32_contents(flat[i].item<float>());
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
  grpc::ChannelArguments ch_args;
  const int max_msg_size = 32 * 1024 * 1024;
  ch_args.SetMaxReceiveMessageSize(max_msg_size);
  ch_args.SetMaxSendMessageSize(max_msg_size);

  auto channel = grpc::CreateCustomChannel(
      "localhost:50051", grpc::InsecureChannelCredentials(), ch_args);

  InferenceClient client(channel);

  client.ServerIsLive();
  client.ModelInfer();
  return 0;
}