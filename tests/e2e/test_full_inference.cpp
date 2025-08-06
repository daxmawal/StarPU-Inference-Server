#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "../test_helpers.hpp"
#include "grpc/server/inference_service.hpp"
#include "grpc_service.grpc.pb.h"

TEST(E2E, FullInference)
{
  namespace fs = std::filesystem;
  auto model_path =
      fs::path(__FILE__).parent_path() / "resources" / "simple_model.ts";
  std::ifstream in(model_path);
  ASSERT_TRUE(in.is_open());
  std::string script(
      (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

  torch::jit::script::Module model("m");
  model.define(script);

  torch::Tensor input = torch::tensor({1.0f, 2.0f});
  std::vector<torch::Tensor> reference_outputs;
  {
    std::vector<torch::IValue> iv{input};
    auto out_iv = model.forward(iv);
    ASSERT_TRUE(out_iv.isTensor());
    reference_outputs.push_back(out_iv.toTensor());
  }

  starpu_server::InferenceQueue queue;

  std::jthread worker([&] {
    std::shared_ptr<starpu_server::InferenceJob> job;
    queue.wait_and_pop(job);
    std::vector<torch::IValue> iv(
        job->get_input_tensors().begin(), job->get_input_tensors().end());
    auto out_iv = model.forward(iv);
    std::vector<torch::Tensor> outs{out_iv.toTensor()};
    job->get_on_complete()(outs, 0.0);
  });

  std::string address = "127.0.0.1:50051";
  auto server =
      starpu_server::start_test_grpc_server(queue, reference_outputs, address);

  auto channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  ASSERT_TRUE(channel->WaitForConnected(
      std::chrono::system_clock::now() + std::chrono::seconds(1)));
  auto stub = inference::GRPCInferenceService::NewStub(channel);

  std::vector<float> in_vals{1.0f, 2.0f};
  auto req = starpu_server::make_model_infer_request(
      {{{2}, at::kFloat, starpu_server::to_raw_data(in_vals)}});
  req.MergeFrom(starpu_server::make_model_request("m", "1"));
  grpc::ClientContext ctx;
  inference::ModelInferResponse resp;
  auto status = stub->ModelInfer(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_GT(resp.server_receive_ms(), 0);
  EXPECT_GT(resp.server_send_ms(), 0);
  starpu_server::verify_populate_response(
      req, resp, reference_outputs, resp.server_receive_ms(),
      resp.server_send_ms());

  starpu_server::StopServer(server.server);
  server.thread.join();
}
