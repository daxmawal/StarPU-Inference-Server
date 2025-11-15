#include <arpa/inet.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <future>
#include <string>
#include <thread>

#include "test_inference_service.hpp"

TEST_F(InferenceServiceTest, ModelInferPropagatesSubmitError)
{
  auto req = starpu_server::make_valid_request();
  req.MergeFrom(starpu_server::make_model_request("m", "1"));
  auto worker = prepare_job({torch::zeros({1})});
  auto status = service->ModelInfer(&ctx, &req, &reply);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
  expect_empty_infer_response(reply);
}

TEST_F(InferenceServiceTest, ModelInferReturnsOutputs)
{
  auto req = starpu_server::make_valid_request();
  req.MergeFrom(starpu_server::make_model_request("m", "1"));
  constexpr float kVal1 = 10.0F;
  constexpr float kVal2 = 20.0F;
  constexpr float kVal3 = 30.0F;
  constexpr float kVal4 = 40.0F;
  std::vector<torch::Tensor> outs = {
      torch::tensor({kVal1, kVal2, kVal3, kVal4}).view({2, 2})};
  auto worker = prepare_job({torch::zeros({2, 2})}, outs);
  auto status = service->ModelInfer(&ctx, &req, &reply);
  ASSERT_TRUE(status.ok());
  EXPECT_GT(reply.server_receive_ms(), 0);
  EXPECT_GT(reply.server_send_ms(), 0);
  auto response_breakdown = starpu_server::make_latency_breakdown(reply);
  starpu_server::verify_populate_response(
      req, reply, outs, reply.server_receive_ms(), reply.server_send_ms(),
      response_breakdown);
  EXPECT_GE(reply.server_total_ms(), 0.0);
}

TEST_F(InferenceServiceTest, ModelInferDetectsInputSizeMismatch)
{
  auto req = starpu_server::make_valid_request();
  req.MergeFrom(starpu_server::make_model_request("m", "1"));
  req.mutable_raw_input_contents(0)->append("0", 1);
  auto status = service->ModelInfer(&ctx, &req, &reply);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  expect_empty_infer_response(reply);
}

TEST_F(InferenceServiceTest, BasicLivenessAndReadiness)
{
  inference::ServerLiveRequest live_req;
  inference::ServerLiveResponse live_resp;
  auto status = service->ServerLive(&ctx, &live_req, &live_resp);
  ASSERT_TRUE(status.ok());
  EXPECT_TRUE(live_resp.live());
  inference::ServerReadyRequest ready_req;
  inference::ServerReadyResponse ready_resp;
  status = service->ServerReady(&ctx, &ready_req, &ready_resp);
  ASSERT_TRUE(status.ok());
  EXPECT_TRUE(ready_resp.ready());
  inference::ModelReadyRequest model_req;
  inference::ModelReadyResponse model_resp;
  status = service->ModelReady(&ctx, &model_req, &model_resp);
  ASSERT_TRUE(status.ok());
  EXPECT_TRUE(model_resp.ready());
}

TEST_P(SubmitJobAndWaitTest, ReturnsExpectedStatus)
{
  std::vector<torch::Tensor> inputs = {torch::tensor({1})};
  std::vector<torch::Tensor> outputs;
  auto worker = prepare_job(GetParam().ref_outputs, GetParam().worker_outputs);
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;
  starpu_server::detail::TimingInfo timing_info{};
  auto status =
      service->submit_job_and_wait(inputs, outputs, breakdown, timing_info);
  EXPECT_EQ(status.error_code(), GetParam().expected_status);
  if (status.ok()) {
    ASSERT_EQ(outputs.size(), GetParam().worker_outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      EXPECT_TRUE(torch::equal(outputs[i], GetParam().worker_outputs[i]));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    SubmitJobAndWaitScenarios, SubmitJobAndWaitTest,
    ::testing::Values(
        SubmitJobAndWaitCase{
            {torch::zeros({1})}, {}, grpc::StatusCode::INTERNAL},
        SubmitJobAndWaitCase{
            {torch::zeros({1})}, {torch::tensor({42})}, grpc::StatusCode::OK}));

TEST(GrpcServer, RunGrpcServer_StartsAndResetsServer)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  std::unique_ptr<grpc::Server> server;
  constexpr std::size_t kMaxMessageSizeMiB = 32U;
  constexpr std::size_t kMiB =
      static_cast<std::size_t>(1024) * static_cast<std::size_t>(1024);
  std::jthread thread([&]() {
    const auto options = starpu_server::GrpcServerOptions{
        "127.0.0.1:0", kMaxMessageSizeMiB * kMiB,
        starpu_server::VerbosityLevel::Info, ""};
    starpu_server::RunGrpcServer(
        queue, reference_outputs, {at::kFloat}, options, server);
  });
  while (!server) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  starpu_server::StopServer(server.get());
  thread.join();
  EXPECT_EQ(server, nullptr);
}

TEST(GrpcServer, RunGrpcServer_WithExpectedDimsResetsServer)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  std::unique_ptr<grpc::Server> server;
  constexpr std::size_t kMaxMessageSizeMiB = 32U;
  constexpr std::size_t kMiB =
      static_cast<std::size_t>(1024) * static_cast<std::size_t>(1024);
  constexpr int kMaxBatchSize = 4;
  const std::vector<at::ScalarType> expected_input_types = {at::kFloat};
  const std::vector<std::vector<int64_t>> expected_input_dims = {
      {kMaxBatchSize, 3, 224, 224}};
  std::jthread thread([&]() {
    const auto options = starpu_server::GrpcServerOptions{
        "127.0.0.1:0", kMaxMessageSizeMiB * kMiB,
        starpu_server::VerbosityLevel::Info, ""};
    starpu_server::RunGrpcServer(
        queue, reference_outputs, expected_input_types, expected_input_dims,
        kMaxBatchSize, options, server);
  });
  while (!server) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  starpu_server::StopServer(server.get());
  thread.join();
  EXPECT_EQ(server, nullptr);
}

TEST(GrpcServer, StartAndStop)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  auto server = starpu_server::start_test_grpc_server(queue, reference_outputs);
  starpu_server::StopServer(server.server.get());
  server.thread.join();
  EXPECT_EQ(server.server, nullptr);
}

TEST(GrpcServer, RunGrpcServer_FailsWhenPortUnavailable)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  std::unique_ptr<grpc::Server> server;
  constexpr std::size_t kMaxMessageSizeMiB = 32U;
  constexpr std::size_t kMiB =
      static_cast<std::size_t>(1024) * static_cast<std::size_t>(1024);

  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(fd, 0);

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;

  ASSERT_EQ(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);

  socklen_t addr_len = sizeof(addr);
  ASSERT_EQ(
      ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &addr_len), 0);
  const int port = ntohs(addr.sin_port);

  const auto endpoint = "127.0.0.1:" + std::to_string(port);
  auto future = std::async(std::launch::async, [&]() {
    const auto options = starpu_server::GrpcServerOptions{
        endpoint, kMaxMessageSizeMiB * kMiB,
        starpu_server::VerbosityLevel::Info, ""};
    starpu_server::RunGrpcServer(
        queue, reference_outputs, {at::kFloat}, options, server);
  });

  EXPECT_EQ(
      future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  future.get();
  EXPECT_EQ(server, nullptr);

  ::close(fd);
}

TEST(GrpcServer, RunGrpcServerWithExpectedDims_FailsWhenPortUnavailable)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  std::unique_ptr<grpc::Server> server;
  constexpr std::size_t kMaxMessageSizeMiB = 32U;
  constexpr std::size_t kMiB =
      static_cast<std::size_t>(1024) * static_cast<std::size_t>(1024);
  constexpr int kMaxBatchSize = 4;
  const std::vector<at::ScalarType> expected_input_types = {at::kFloat};
  const std::vector<std::vector<int64_t>> expected_input_dims = {
      {kMaxBatchSize, 3, 224, 224}};

  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(fd, 0);

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;

  ASSERT_EQ(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);

  socklen_t addr_len = sizeof(addr);
  ASSERT_EQ(
      ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &addr_len), 0);
  const int port = ntohs(addr.sin_port);

  const auto endpoint = "127.0.0.1:" + std::to_string(port);
  auto future = std::async(std::launch::async, [&]() {
    const auto options = starpu_server::GrpcServerOptions{
        endpoint, kMaxMessageSizeMiB * kMiB,
        starpu_server::VerbosityLevel::Info, ""};
    starpu_server::RunGrpcServer(
        queue, reference_outputs, expected_input_types, expected_input_dims,
        kMaxBatchSize, options, server);
  });

  EXPECT_EQ(
      future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  future.get();
  EXPECT_EQ(server, nullptr);

  ::close(fd);
}

namespace {

auto
pick_unused_port() -> int
{
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;

  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return -1;
  }

  socklen_t addr_len = sizeof(addr);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &addr_len) != 0) {
    ::close(fd);
    return -1;
  }

  const int port = ntohs(addr.sin_port);
  ::close(fd);
  return port;
}

}  // namespace

TEST(GrpcServer, RunGrpcServerProcessesUnaryRequest)
{
  const int port = pick_unused_port();
  ASSERT_GT(port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(port);

  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  std::unique_ptr<grpc::Server> server;

  constexpr std::size_t kMaxMessageSizeMiB = 32U;
  constexpr std::size_t kMiB =
      static_cast<std::size_t>(1024) * static_cast<std::size_t>(1024);

  const auto options = starpu_server::GrpcServerOptions{
      address, kMaxMessageSizeMiB * kMiB, starpu_server::VerbosityLevel::Info,
      ""};

  std::jthread thread([&, options]() {
    starpu_server::RunGrpcServer(
        queue, reference_outputs, {at::kFloat}, options, server);
  });

  while (!server) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  auto channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  ASSERT_TRUE(channel->WaitForConnected(
      std::chrono::system_clock::now() + std::chrono::seconds(5)));

  auto stub = inference::GRPCInferenceService::NewStub(channel);

  grpc::ClientContext context;
  inference::ServerLiveRequest request;
  inference::ServerLiveResponse response;
  const auto status = stub->ServerLive(&context, request, &response);
  ASSERT_TRUE(status.ok());
  EXPECT_TRUE(response.live());

  starpu_server::StopServer(server.get());
  thread.join();
  EXPECT_EQ(server, nullptr);
}

TEST(GrpcServer, RunGrpcServerProcessesModelInferRequest)
{
  const int port = pick_unused_port();
  ASSERT_GT(port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(port);

  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs = {torch::zeros({2, 2})};
  std::unique_ptr<grpc::Server> server;

  constexpr std::size_t kMaxMessageSizeMiB = 32U;
  constexpr std::size_t kMiB =
      static_cast<std::size_t>(1024) * static_cast<std::size_t>(1024);

  const auto options = starpu_server::GrpcServerOptions{
      address, kMaxMessageSizeMiB * kMiB, starpu_server::VerbosityLevel::Info,
      ""};

  constexpr float kVal1 = 10.0F;
  constexpr float kVal2 = 20.0F;
  constexpr float kVal3 = 30.0F;
  constexpr float kVal4 = 40.0F;
  std::vector<torch::Tensor> expected_outputs = {
      torch::tensor({kVal1, kVal2, kVal3, kVal4}).view({2, 2})};

  auto worker = starpu_server::run_single_job(queue, expected_outputs);

  std::jthread thread([&, options]() {
    starpu_server::RunGrpcServer(
        queue, reference_outputs, {at::kFloat}, options, server);
  });

  while (!server) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  auto channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  ASSERT_TRUE(channel->WaitForConnected(
      std::chrono::system_clock::now() + std::chrono::seconds(5)));

  auto stub = inference::GRPCInferenceService::NewStub(channel);

  auto request = starpu_server::make_valid_request();
  request.MergeFrom(starpu_server::make_model_request("model", "1"));

  inference::ModelInferResponse response;
  grpc::ClientContext context;
  const auto status = stub->ModelInfer(&context, request, &response);
  ASSERT_TRUE(status.ok());

  EXPECT_GT(response.server_receive_ms(), 0);
  EXPECT_GT(response.server_send_ms(), 0);
  auto response_breakdown = starpu_server::make_latency_breakdown(response);
  starpu_server::verify_populate_response(
      request, response, expected_outputs, response.server_receive_ms(),
      response.server_send_ms(), response_breakdown);
  EXPECT_GE(response.server_total_ms(), 0.0);

  worker.join();

  starpu_server::StopServer(server.get());
  thread.join();
  EXPECT_EQ(server, nullptr);
}
