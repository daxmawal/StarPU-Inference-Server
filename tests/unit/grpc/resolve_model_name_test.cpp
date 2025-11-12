#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

#define private public
#include "grpc/server/inference_service.hpp"
#undef private

namespace {

class ResolveModelNameTest : public ::testing::Test {
 protected:
  [[nodiscard]] auto make_service(std::string default_model_name)
      -> std::unique_ptr<starpu_server::InferenceServiceImpl>
  {
    return std::make_unique<starpu_server::InferenceServiceImpl>(
        &queue_, &reference_outputs_, std::vector<at::ScalarType>{at::kFloat},
        std::move(default_model_name));
  }

  starpu_server::InferenceQueue queue_;
  std::vector<torch::Tensor> reference_outputs_ = {
      torch::zeros({1}, torch::TensorOptions().dtype(at::kFloat))};
};

TEST_F(ResolveModelNameTest, ReturnsDefaultNameWhenClientRequestsDifferentModel)
{
  auto service = make_service("server_default");
  const auto resolved = service->resolve_model_name("client_provided");
  EXPECT_EQ(resolved, "server_default");
}

TEST_F(
    ResolveModelNameTest,
    ReturnsDefaultNameWhenClientRequestDoesNotSpecifyModel)
{
  auto service = make_service("server_default");
  const auto resolved = service->resolve_model_name("");
  EXPECT_EQ(resolved, "server_default");
}

}  // namespace
