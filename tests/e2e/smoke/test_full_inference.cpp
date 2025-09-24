#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "grpc/server/inference_service.hpp"
#include "grpc_service.grpc.pb.h"
#include "test_helpers.hpp"

namespace {
constexpr float kVal1 = 1.0F;
constexpr float kVal2 = 2.0F;

auto
start_worker(
    starpu_server::InferenceQueue& queue,
    torch::jit::script::Module& model) -> std::jthread
{
  return std::jthread([&] {
    std::shared_ptr<starpu_server::InferenceJob> job;
    if (!queue.wait_and_pop(job)) {
      return;
    }
    std::vector<torch::IValue> value(
        job->get_input_tensors().begin(), job->get_input_tensors().end());
    auto out_value = model.forward(value);
    std::vector<torch::Tensor> outs{out_value.toTensor()};
    job->get_on_complete()(outs, 0.0);
  });
}

void
expect_connected(const std::shared_ptr<grpc::Channel>& channel)
{
  ASSERT_TRUE(channel->WaitForConnected(
      std::chrono::system_clock::now() + std::chrono::seconds(1)));
}
}  // namespace

TEST(E2E, FullInference)
{
  namespace fs = std::filesystem;
  auto model_path =
      fs::path(__FILE__).parent_path() / ".." / "fixtures" / "simple_model.ts";
  std::ifstream stream(model_path);
  ASSERT_TRUE(stream.is_open());
  std::string script(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());

  torch::jit::script::Module model("m");
  model.define(script);

  torch::Tensor input = torch::tensor({kVal1, kVal2});
  std::vector<torch::Tensor> reference_outputs;
  {
    std::vector<torch::IValue> value{input};
    auto out_value = model.forward(value);
    ASSERT_TRUE(out_value.isTensor());
    reference_outputs.push_back(out_value.toTensor());
  }

  starpu_server::InferenceQueue queue;

  auto worker = start_worker(queue, model);

  auto server = starpu_server::start_test_grpc_server(queue, reference_outputs);
  std::string address = "127.0.0.1:" + std::to_string(server.port);
  auto channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  expect_connected(channel);
  auto stub = inference::GRPCInferenceService::NewStub(channel);

  std::vector<float> in_vals{kVal1, kVal2};
  auto req = starpu_server::make_model_infer_request(
      {{{2}, at::kFloat, starpu_server::to_raw_data(in_vals)}});
  req.MergeFrom(starpu_server::make_model_request("m", "1"));
  grpc::ClientContext ctx;
  inference::ModelInferResponse resp;
  auto status = stub->ModelInfer(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_GT(resp.server_receive_ms(), 0);
  EXPECT_GT(resp.server_send_ms(), 0);
  starpu_server::InferenceServiceImpl::LatencyBreakdown response_breakdown;
  response_breakdown.preprocess_ms = resp.server_preprocess_ms();
  response_breakdown.queue_ms = resp.server_queue_ms();
  response_breakdown.submit_ms = resp.server_submit_ms();
  response_breakdown.scheduling_ms = resp.server_scheduling_ms();
  response_breakdown.codelet_ms = resp.server_codelet_ms();
  response_breakdown.inference_ms = resp.server_inference_ms();
  response_breakdown.callback_ms = resp.server_callback_ms();
  response_breakdown.postprocess_ms = resp.server_postprocess_ms();
  response_breakdown.total_ms = resp.server_total_ms();
  response_breakdown.overall_ms = resp.server_overall_ms();
  starpu_server::verify_populate_response(
      req, resp, reference_outputs, resp.server_receive_ms(),
      resp.server_send_ms(), response_breakdown);
  EXPECT_GE(resp.server_total_ms(), 0.0);

  starpu_server::StopServer(server.server);
  server.thread.join();
}
