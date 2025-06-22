#include <grpcpp/grpcpp.h>
#include <torch/script.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "client_args.hpp"
#include "grpc_service.grpc.pb.h"
#include "utils/logger.hpp"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using inference::GRPCInferenceService;
using inference::ModelInferRequest;
using inference::ModelInferResponse;
using inference::ServerLiveRequest;
using inference::ServerLiveResponse;
using inference::ServerReadyRequest;
using inference::ServerReadyResponse;

constexpr int MillisecondsPerSecond = 1000;

struct AsyncClientCall {
  int request_id = 0;
  ModelInferResponse reply;
  ClientContext context;
  Status status;
  std::unique_ptr<grpc::ClientAsyncResponseReader<ModelInferResponse>>
      response_reader = nullptr;
  std::chrono::high_resolution_clock::time_point start_time;
};

auto
FormatTimestamp(const std::chrono::high_resolution_clock::time_point&
                    time_point) -> std::string
{
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                          time_point.time_since_epoch()) %
                      MillisecondsPerSecond;

  std::time_t time = std::chrono::system_clock::to_time_t(
      std::chrono::time_point_cast<std::chrono::system_clock::duration>(
          time_point));
  std::tm local_tm{};
  localtime_r(&time, &local_tm);
  std::ostringstream oss;
  oss << std::put_time(&local_tm, "%H:%M:%S") << '.' << std::setfill('0')
      << std::setw(3) << milliseconds.count();
  return oss.str();
}

class InferenceClient {
 public:
  explicit InferenceClient(
      std::shared_ptr<Channel>& channel, VerbosityLevel verbosity)
      : stub_(GRPCInferenceService::NewStub(channel)), verbosity_(verbosity)
  {
  }

  auto ServerIsLive() -> bool
  {
    const ServerLiveRequest request;
    ServerLiveResponse response;
    ClientContext context;

    Status status = stub_->ServerLive(&context, request, &response);

    if (!status.ok()) {
      log_error("RPC failed: " + status.error_message());
      return false;
    }

    log_info(
        verbosity_,
        std::string("Server live: ") + (response.live() ? "true" : "false"));
    return response.live();
  }

  auto ServerIsReady() -> bool
  {
    const ServerReadyRequest request;
    ServerReadyResponse response;
    ClientContext context;

    Status status = stub_->ServerReady(&context, request, &response);

    if (!status.ok()) {
      std::cerr << "RPC failed: " << status.error_message() << std::endl;
      return false;
    }

    log_info(
        verbosity_,
        std::string("Server ready: ") + (response.ready() ? "true" : "false"));
    return response.ready();
  }

  void AsyncModelInfer(const torch::Tensor& tensor, const ClientConfig& cfg)
  {
    const int current_id = next_request_id_++;

    auto call = std::make_unique<AsyncClientCall>();
    call->request_id = current_id;
    call->start_time = std::chrono::high_resolution_clock::now();

    log_info(verbosity_, "Sending request ID: " + std::to_string(current_id));

    ModelInferRequest request;
    request.set_model_name(cfg.model_name);
    request.set_model_version(cfg.model_version);
    request.set_client_send_ms(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            call->start_time.time_since_epoch())
            .count());

    auto* input = request.add_inputs();
    input->set_name("input");
    input->set_datatype(scalar_type_to_string(cfg.type));
    for (auto dim : cfg.shape) {
      input->add_shape(dim);
    }

    auto flat = tensor.view({-1});
    request.add_raw_input_contents()->assign(
        reinterpret_cast<const char*>(flat.data_ptr()),
        flat.numel() * flat.element_size());

    call->response_reader =
        stub_->AsyncModelInfer(&call->context, request, &cq_);
    call->response_reader->Finish(&call->reply, &call->status, call.get());
    call.release();
  }

  void AsyncCompleteRpc()
  {
    void* got_tag = nullptr;
    bool call_ctx = false;
    while (cq_.Next(&got_tag, &call_ctx)) {
      if (!call_ctx) {
        log_warning("Received invalid RPC completion, exiting CQ loop");
        break;
      }
      std::unique_ptr<AsyncClientCall> call(
          static_cast<AsyncClientCall*>(got_tag));
      auto end = std::chrono::high_resolution_clock::now();
      auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
                         end - call->start_time)
                         .count();

      auto sent_time_str = FormatTimestamp(call->start_time);
      auto recv_time_str = FormatTimestamp(end);

      if (call->status.ok()) {
        auto start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            call->start_time.time_since_epoch())
                            .count();
        auto request_tx = call->reply.server_receive_ms() - start_ms;
        auto response_tx =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                end.time_since_epoch())
                .count() -
            call->reply.server_send_ms();

        log_info(
            verbosity_,
            "Request ID " + std::to_string(call->request_id) + " sent at " +
                sent_time_str + ", received at " + recv_time_str +
                ", latency: " + std::to_string(latency) + " ms" +
                ", req_tx: " + std::to_string(request_tx) + " ms" +
                ", resp_tx: " + std::to_string(response_tx) + " ms");
      } else {
        log_error(
            "Request ID " + std::to_string(call->request_id) + " failed at " +
            recv_time_str + ": " + call->status.error_message());
      }
    }
  }


  void Shutdown() { cq_.Shutdown(); }

 private:
  std::unique_ptr<GRPCInferenceService::Stub> stub_;
  grpc::CompletionQueue cq_;
  std::atomic<int> next_request_id_{0};
  VerbosityLevel verbosity_;
};

auto
main(int argc, char* argv[]) -> int
{
  std::vector<const char*> const_argv(argv, argv + argc);
  std::span<const char*> args{const_argv};
  const ClientConfig config = parse_client_args(args);
  if (config.show_help) {
    display_client_help(args.front());
    return 0;
  }
  if (!config.valid) {
    log_error("Invalid program options.");
    return 1;
  }
  grpc::ChannelArguments ch_args;
  const int max_msg_size = 32 * 1024 * 1024;
  ch_args.SetMaxReceiveMessageSize(max_msg_size);
  ch_args.SetMaxSendMessageSize(max_msg_size);

  auto channel = grpc::CreateCustomChannel(
      config.server_address, grpc::InsecureChannelCredentials(), ch_args);

  InferenceClient client(channel, config.verbosity);

  if (!client.ServerIsLive()) {
    return 1;
  }

  if (!client.ServerIsReady()) {
    return 1;
  }

  constexpr int NUM_TENSORS = 5;
  std::vector<torch::Tensor> tensor_pool;
  tensor_pool.reserve(NUM_TENSORS);
  for (int i = 0; i < NUM_TENSORS; ++i) {
    tensor_pool.push_back(
        torch::rand(config.shape, torch::TensorOptions().dtype(config.type)));
  }

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, NUM_TENSORS - 1);

  std::jthread cq_thread(&InferenceClient::AsyncCompleteRpc, &client);
  for (int i = 0; i < config.iterations; ++i) {
    const auto& tensor = tensor_pool[dist(rng)];
    client.AsyncModelInfer(tensor, config);
    std::this_thread::sleep_for(std::chrono::milliseconds(config.delay_ms));
  }

  client.Shutdown();

  return 0;
}