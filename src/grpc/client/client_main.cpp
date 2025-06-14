#include <grpcpp/grpcpp.h>
#include <torch/script.h>

#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "grpc_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using inference::GRPCInferenceService;
using inference::ModelInferRequest;
using inference::ModelInferResponse;
using inference::ServerLiveRequest;
using inference::ServerLiveResponse;

constexpr int BATCH_SIZE = 1;
constexpr int CHANNELS = 3;
constexpr int HEIGHT = 224;
constexpr int WIDTH = 224;

struct AsyncClientCall {
  int request_id;
  ModelInferResponse reply;
  ClientContext context;
  Status status;
  std::unique_ptr<grpc::ClientAsyncResponseReader<ModelInferResponse>>
      response_reader;
  std::chrono::high_resolution_clock::time_point start_time;
};

std::string
FormatTimestamp(const std::chrono::high_resolution_clock::time_point& tp)
{
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                tp.time_since_epoch()) %
            1000;

  std::time_t t = std::chrono::system_clock::to_time_t(
      std::chrono::time_point_cast<std::chrono::system_clock::duration>(tp));
  std::tm tm{};
  localtime_r(&t, &tm);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%H:%M:%S") << '.' << std::setfill('0')
      << std::setw(3) << ms.count();
  return oss.str();
}

class InferenceClient {
 public:
  explicit InferenceClient(std::shared_ptr<Channel>& channel)
      : stub_(GRPCInferenceService::NewStub(channel))
  {
  }

  auto ServerIsLive() -> bool
  {
    ServerLiveRequest request;
    ServerLiveResponse response;
    ClientContext context;

    Status status = stub_->ServerLive(&context, request, &response);

    if (!status.ok()) {
      std::cerr << "RPC failed: " << status.error_message() << std::endl;
      return false;
    }

    std::cout << "Server live: " << std::boolalpha << response.live()
              << std::endl;
    return response.live();
  }

  void AsyncModelInfer(const torch::Tensor& tensor)
  {
    int current_id = next_request_id_++;

    std::cout << "Sending request ID: " << current_id << std::endl;

    ModelInferRequest request;
    request.set_model_name("example");
    request.set_model_version("1");

    auto* input = request.add_inputs();
    input->set_name("input");
    input->set_datatype("FP32");
    input->add_shape(BATCH_SIZE);
    input->add_shape(CHANNELS);
    input->add_shape(HEIGHT);
    input->add_shape(WIDTH);

    auto flat = tensor.view({-1});
    auto* contents = input->mutable_contents();
    contents->mutable_fp32_contents()->Reserve(flat.numel());
    for (int64_t i = 0; i < flat.numel(); ++i) {
      contents->add_fp32_contents(flat[i].item<float>());
    }

    auto* call = new AsyncClientCall;
    call->request_id = current_id;
    call->start_time = std::chrono::high_resolution_clock::now();
    call->response_reader =
        stub_->AsyncModelInfer(&call->context, request, &cq_);
    call->response_reader->Finish(&call->reply, &call->status, call);
  }

  void AsyncCompleteRpc()
  {
    void* got_tag = nullptr;
    bool ok = false;
    while (cq_.Next(&got_tag, &ok)) {
      auto* call = static_cast<AsyncClientCall*>(got_tag);
      auto end = std::chrono::high_resolution_clock::now();
      auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
                         end - call->start_time)
                         .count();

      auto sent_time_str = FormatTimestamp(call->start_time);
      auto recv_time_str = FormatTimestamp(end);

      if (call->status.ok()) {
        std::cout << "Request ID " << call->request_id << " sent at "
                  << sent_time_str << ", received at " << recv_time_str
                  << ", latency: " << latency << " ms" << std::endl;
      } else {
        std::cerr << "Request ID " << call->request_id << " failed at "
                  << recv_time_str << ": " << call->status.error_message()
                  << std::endl;
      }
      delete call;
    }
  }


  void Shutdown() { cq_.Shutdown(); }

 private:
  std::unique_ptr<GRPCInferenceService::Stub> stub_;
  grpc::CompletionQueue cq_;
  std::atomic<int> next_request_id_;
};

auto
main() -> int
{
  grpc::ChannelArguments ch_args;
  const int max_msg_size = 32 * 1024 * 1024;
  ch_args.SetMaxReceiveMessageSize(max_msg_size);
  ch_args.SetMaxSendMessageSize(max_msg_size);

  auto channel = grpc::CreateCustomChannel(
      "localhost:50051", grpc::InsecureChannelCredentials(), ch_args);

  InferenceClient client(channel);

  if (!client.ServerIsLive()) {
    return 1;
  }

  constexpr int NUM_TENSORS = 5;
  std::vector<torch::Tensor> tensor_pool;
  tensor_pool.reserve(NUM_TENSORS);
  for (int i = 0; i < NUM_TENSORS; ++i) {
    tensor_pool.push_back(
        torch::rand({BATCH_SIZE, CHANNELS, HEIGHT, WIDTH}, torch::kFloat32));
  }

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, NUM_TENSORS - 1);

  constexpr int NUM_REQUESTS = 100;
  std::thread cq_thread(&InferenceClient::AsyncCompleteRpc, &client);
  for (int i = 0; i < NUM_REQUESTS; ++i) {
    const auto& t = tensor_pool[dist(rng)];
    client.AsyncModelInfer(t);
    std::this_thread::sleep_for(std::chrono::milliseconds(0));
  }

  client.Shutdown();
  cq_thread.join();

  return 0;
}