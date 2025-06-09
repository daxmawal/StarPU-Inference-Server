#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>
#include <string>

#include "inference.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using inference::InferenceRequest;
using inference::InferenceResponse;
using inference::InferenceService;

class InferenceClient {
 public:
  InferenceClient(std::shared_ptr<Channel> channel)
      : stub_(InferenceService::NewStub(channel))
  {
  }

  std::string RunInference(const std::string& tensor_data)
  {
    InferenceRequest request;
    request.set_tensor_data(tensor_data);

    InferenceResponse response;
    ClientContext context;

    Status status = stub_->RunInference(&context, request, &response);

    if (status.ok()) {
      return response.tensor_data();
    } else {
      std::cerr << "RPC failed: " << status.error_message() << std::endl;
      return "";
    }
  }

 private:
  std::unique_ptr<InferenceService::Stub> stub_;
};

int
main()
{
  InferenceClient client(grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials()));

  std::string input = "example tensor data";
  std::string output = client.RunInference(input);

  std::cout << "Inference response: " << output << std::endl;

  return 0;
}