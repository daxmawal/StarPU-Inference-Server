#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <vector>

#include "grpc/server/inference_service.hpp"

namespace {

class AsyncServerContextFixture : public ::testing::Test {
 protected:
  AsyncServerContextFixture() : impl(&queue, &reference_outputs, expected_types)
  {
  }

  inference::GRPCInferenceService::AsyncService async_service;
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  std::vector<at::ScalarType> expected_types;
  starpu_server::InferenceServiceImpl impl;
};

}  // namespace

TEST_F(AsyncServerContextFixture, StartWithoutConfigureDoesNotLaunch)
{
  starpu_server::AsyncServerContext context(async_service, impl);

  context.start();

  EXPECT_FALSE(context.started());
  EXPECT_EQ(context.thread_count(), 0U);
}

TEST_F(AsyncServerContextFixture, StartAfterConfigureIsIdempotent)
{
  starpu_server::AsyncServerContext context(async_service, impl);

  grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials());
  context.configure(builder);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);

  context.start();
  const bool initial_started = context.started();
  const auto initial_thread_count = context.thread_count();

  context.start();

  EXPECT_EQ(context.started(), initial_started);
  EXPECT_EQ(context.thread_count(), initial_thread_count);

  server->Shutdown();
  context.shutdown();
  server->Wait();
}
