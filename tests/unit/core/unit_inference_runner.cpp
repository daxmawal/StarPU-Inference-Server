#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "core/inference_runner.hpp"
#include "core/tensor_builder.hpp"
#include "test_inference_runner.hpp"


namespace {
constexpr int64_t kJobId = 7;
constexpr int64_t kWorkerId = 3;
constexpr double kLatencyMs = 123.0;

const std::vector<int64_t> kShape2x2{2, 2};
const std::vector<int64_t> kShape2x3{2, 3};
const std::vector<int64_t> kShape1{1};

const std::vector<torch::Dtype> kTypesFloatInt{torch::kFloat32, torch::kInt64};
}  // namespace

TEST(InferenceRunner_Unit, MakeShutdownJob)
{
  auto job = starpu_server::InferenceJob::make_shutdown_job();
  ASSERT_NE(job, nullptr);
  EXPECT_TRUE(job->is_shutdown());
}

TEST(InferenceJob_Unit, SettersGettersAndCallback)
{
  const std::vector<torch::Tensor> inputs{torch::ones(kShape2x2)};
  const std::vector<at::ScalarType> types{at::kFloat};
  const std::vector<torch::Tensor> outputs{torch::zeros(kShape2x2)};

  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_job_id(kJobId);
  job->set_input_tensors(inputs);
  job->set_input_types(types);
  job->set_outputs_tensors(outputs);
  job->set_fixed_worker_id(kWorkerId);

  const auto start = std::chrono::high_resolution_clock::now();
  job->set_start_time(start);

  bool callback_called = false;
  std::vector<torch::Tensor> cb_tensors;
  double cb_latency = 0.0;

  job->set_on_complete([&](std::vector<torch::Tensor> tensor, double lat) {
    callback_called = true;
    cb_tensors = std::move(tensor);
    cb_latency = lat;
  });

  EXPECT_EQ(job->get_job_id(), kJobId);
  ASSERT_EQ(job->get_input_tensors().size(), 1U);
  EXPECT_TRUE(job->get_input_tensors()[0].equal(inputs[0]));
  ASSERT_EQ(job->get_input_types().size(), 1U);
  EXPECT_EQ(job->get_input_types()[0], at::kFloat);
  ASSERT_EQ(job->get_output_tensors().size(), 1U);
  EXPECT_TRUE(job->get_output_tensors()[0].equal(outputs[0]));
  EXPECT_EQ(job->get_start_time(), start);
  ASSERT_TRUE(job->get_fixed_worker_id().has_value());
  EXPECT_EQ(job->get_fixed_worker_id().value(), kWorkerId);
  ASSERT_TRUE(job->has_on_complete());

  job->get_on_complete()(job->get_output_tensors(), kLatencyMs);
  EXPECT_TRUE(callback_called);
  ASSERT_EQ(cb_tensors.size(), 1U);
  EXPECT_TRUE(cb_tensors[0].equal(outputs[0]));
  EXPECT_DOUBLE_EQ(cb_latency, kLatencyMs);
}

TEST(InferenceRunnerUtils_Unit, GenerateInputsShapeAndType)
{
  const std::vector<std::vector<int64_t>> shapes{kShape2x3, kShape1};
  std::vector<starpu_server::TensorConfig> cfgs;
  cfgs.reserve(shapes.size());
  for (size_t i = 0; i < shapes.size(); ++i) {
    cfgs.push_back({
        std::string("input") + std::to_string(i), shapes[i], kTypesFloatInt[i]});
  }
  torch::manual_seed(0);
  auto tensors = starpu_server::generate_inputs(cfgs);
  ASSERT_EQ(tensors.size(), 2U);
  EXPECT_EQ(
      tensors[0].sizes(), (torch::IntArrayRef{kShape2x3[0], kShape2x3[1]}));
  EXPECT_EQ(tensors[0].dtype(), torch::kFloat32);
  EXPECT_EQ(tensors[1].sizes(), (torch::IntArrayRef{kShape1[0]}));
  EXPECT_EQ(tensors[1].dtype(), torch::kInt64);
}

TEST(RunInference_Unit, CopyOutputToBufferCopiesData)
{
  auto tensor =
      torch::tensor({1.0F, 2.0F, 3.5F, -4.0F, 0.25F}, torch::kFloat32);
  std::vector<float> dst(5, 0.0F);
  starpu_server::TensorBuilder::copy_output_to_buffer(
      tensor, dst.data(), tensor.numel());
  ASSERT_EQ(dst.size(), 5U);
  EXPECT_FLOAT_EQ(dst[0], 1.0F);
  EXPECT_FLOAT_EQ(dst[1], 2.0F);
  EXPECT_FLOAT_EQ(dst[2], 3.5F);
  EXPECT_FLOAT_EQ(dst[3], -4.0F);
  EXPECT_FLOAT_EQ(dst[4], 0.25F);
}
