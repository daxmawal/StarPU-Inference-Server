#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "grpc_service.grpc.pb.h"
#include "test_helpers.hpp"

namespace {
constexpr float kVal1 = 1.0F;
constexpr float kVal2 = 2.0F;

void
expect_connected(const std::shared_ptr<grpc::Channel>& channel)
{
  ASSERT_TRUE(channel->WaitForConnected(
      std::chrono::system_clock::now() + std::chrono::seconds(2)));
}

auto
load_simple_model() -> torch::jit::script::Module
{
  namespace fs = std::filesystem;
  const auto model_path =
      fs::path(__FILE__).parent_path() / ".." / "fixtures" / "simple_model.ts";

  std::ifstream stream(model_path);
  EXPECT_TRUE(stream.is_open());
  std::string script(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());

  torch::jit::script::Module model("m");
  model.define(script);
  return model;
}

auto
build_reference_outputs(torch::jit::script::Module& model)
    -> std::vector<torch::Tensor>
{
  const torch::Tensor input = torch::tensor({kVal1, kVal2});
  std::vector<torch::IValue> value{input};
  auto out_value = model.forward(value);
  EXPECT_TRUE(out_value.isTensor());
  return {out_value.toTensor()};
}

auto
make_request() -> inference::ModelInferRequest
{
  std::vector<float> in_vals{kVal1, kVal2};
  auto req = starpu_server::make_model_infer_request(
      {{{2}, at::kFloat, starpu_server::to_raw_data(in_vals)}});
  req.MergeFrom(starpu_server::make_model_request("m", "1"));
  return req;
}
}  // namespace

TEST(E2ERegression, QueueFullUnderConcurrentLoadReturnsResourceExhausted)
{
  auto model = load_simple_model();
  auto reference_outputs = build_reference_outputs(model);

  starpu_server::InferenceQueue queue(/*max_size=*/1);
  auto server = starpu_server::start_test_grpc_server(queue, reference_outputs);
  const std::string address = "127.0.0.1:" + std::to_string(server.port);
  auto channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  expect_connected(channel);

  std::promise<grpc::Status> first_status_promise;
  auto first_status_future = first_status_promise.get_future();
  inference::ModelInferResponse first_response;
  const auto first_request = make_request();
  std::jthread first_client([&, first_request]() mutable {
    auto stub = inference::GRPCInferenceService::NewStub(channel);
    grpc::ClientContext ctx;
    ctx.set_deadline(
        std::chrono::system_clock::now() + std::chrono::seconds(3));
    first_status_promise.set_value(
        stub->ModelInfer(&ctx, first_request, &first_response));
  });

  const auto queue_filled_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (queue.size() < 1U &&
         std::chrono::steady_clock::now() < queue_filled_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  ASSERT_EQ(queue.size(), 1U)
      << "The first request did not fill the queue before load started";

  constexpr int kClientThreads = 6;
  constexpr int kRequestsPerThread = 10;
  constexpr int kExpectedRejections = kClientThreads * kRequestsPerThread;

  std::atomic<int> rejected_requests{0};
  std::atomic<int> unexpected_statuses{0};
  std::vector<std::jthread> clients;
  clients.reserve(kClientThreads);
  for (int i = 0; i < kClientThreads; ++i) {
    clients.emplace_back([&]() {
      auto stub = inference::GRPCInferenceService::NewStub(channel);
      const auto request = make_request();
      for (int req_idx = 0; req_idx < kRequestsPerThread; ++req_idx) {
        grpc::ClientContext ctx;
        ctx.set_deadline(
            std::chrono::system_clock::now() + std::chrono::milliseconds(250));
        inference::ModelInferResponse response;
        const auto status = stub->ModelInfer(&ctx, request, &response);
        if (status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED) {
          rejected_requests.fetch_add(1, std::memory_order_relaxed);
        } else {
          unexpected_statuses.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  clients.clear();

  std::shared_ptr<starpu_server::InferenceJob> blocked_job;
  ASSERT_TRUE(queue.wait_for_and_pop(blocked_job, std::chrono::seconds(1)));
  ASSERT_NE(blocked_job, nullptr);
  auto outputs_copy = reference_outputs;
  blocked_job->completion().get_on_complete()(outputs_copy, 0.0);

  ASSERT_EQ(
      first_status_future.wait_for(std::chrono::seconds(2)),
      std::future_status::ready);
  const auto first_status = first_status_future.get();
  ASSERT_TRUE(first_status.ok()) << first_status.error_message();
  EXPECT_GT(first_response.server_receive_ms(), 0);
  EXPECT_GT(first_response.server_send_ms(), 0);

  EXPECT_EQ(
      rejected_requests.load(std::memory_order_relaxed), kExpectedRejections);
  EXPECT_EQ(unexpected_statuses.load(std::memory_order_relaxed), 0);

  starpu_server::StopServer(server.server.get());
  server.thread.join();
}
