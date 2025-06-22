#include "inference_client.hpp"

#include "utils/time_utils.hpp"

struct AsyncClientCall {
  int request_id = 0;
  inference::ModelInferResponse reply;
  grpc::ClientContext context;
  grpc::Status status;
  std::unique_ptr<
      grpc::ClientAsyncResponseReader<inference::ModelInferResponse>>
      response_reader = nullptr;
  std::chrono::high_resolution_clock::time_point start_time;
};

InferenceClient::InferenceClient(
    std::shared_ptr<grpc::Channel>& channel, VerbosityLevel verbosity)
    : stub_(inference::GRPCInferenceService::NewStub(channel)),
      verbosity_(verbosity)
{
}

auto
InferenceClient::ServerIsLive() -> bool
{
  const inference::ServerLiveRequest request;
  inference::ServerLiveResponse response;
  grpc::ClientContext context;

  grpc::Status status = stub_->ServerLive(&context, request, &response);

  if (!status.ok()) {
    log_error("RPC failed: " + status.error_message());
    return false;
  }

  log_info(
      verbosity_,
      std::string("Server live: ") + (response.live() ? "true" : "false"));
  return response.live();
}

auto
InferenceClient::ServerIsReady() -> bool
{
  const inference::ServerReadyRequest request;
  inference::ServerReadyResponse response;
  grpc::ClientContext context;

  grpc::Status status = stub_->ServerReady(&context, request, &response);

  if (!status.ok()) {
    std::cerr << "RPC failed: " << status.error_message() << std::endl;
    return false;
  }

  log_info(
      verbosity_,
      std::string("Server ready: ") + (response.ready() ? "true" : "false"));
  return response.ready();
}

auto
InferenceClient::ModelIsReady(
    const std::string& name, const std::string& version) -> bool
{
  inference::ModelReadyRequest request;
  request.set_name(name);
  request.set_version(version);
  inference::ModelReadyResponse response;
  grpc::ClientContext context;

  grpc::Status status = stub_->ModelReady(&context, request, &response);

  if (!status.ok()) {
    log_error("RPC failed: " + status.error_message());
    return false;
  }

  log_info(
      verbosity_,
      std::string("Model ready: ") + (response.ready() ? "true" : "false"));
  return response.ready();
}

void
InferenceClient::AsyncModelInfer(
    const torch::Tensor& tensor, const ClientConfig& cfg)
{
  const int current_id = next_request_id_++;

  auto call = std::make_unique<AsyncClientCall>();
  call->request_id = current_id;
  call->start_time = std::chrono::high_resolution_clock::now();

  log_info(verbosity_, "Sending request ID: " + std::to_string(current_id));

  inference::ModelInferRequest request;
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

  call->response_reader = stub_->AsyncModelInfer(&call->context, request, &cq_);
  call->response_reader->Finish(&call->reply, &call->status, call.get());
  call.release();
}

void
InferenceClient::AsyncCompleteRpc()
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

    auto sent_time_str = time_utils::format_timestamp(call->start_time);
    auto recv_time_str = time_utils::format_timestamp(end);

    if (call->status.ok()) {
      auto start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          call->start_time.time_since_epoch())
                          .count();
      auto request_tx = call->reply.server_receive_ms() - start_ms;
      auto response_tx = std::chrono::duration_cast<std::chrono::milliseconds>(
                             end.time_since_epoch())
                             .count() -
                         call->reply.server_send_ms();

      log_info(
          verbosity_, "Request ID " + std::to_string(call->request_id) +
                          " sent at " + sent_time_str + ", received at " +
                          recv_time_str +
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

void
InferenceClient::Shutdown()
{
  cq_.Shutdown();
}