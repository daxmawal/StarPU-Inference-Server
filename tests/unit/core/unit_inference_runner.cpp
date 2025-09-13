#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "core/inference_runner.hpp"
#include "core/tensor_builder.hpp"
#include "test_helpers.hpp"
#include "test_inference_runner.hpp"

namespace {
inline auto
cuda_sync_result_ref() -> cudaError_t&
{
  static cudaError_t value = cudaSuccess;
  return value;
}
inline void
SetCudaSyncResult(cudaError_t status)
{
  cuda_sync_result_ref() = status;
}
}  // namespace

extern "C" auto
cudaDeviceSynchronize() -> cudaError_t
{
  return cuda_sync_result_ref();
}

extern "C" auto
cudaGetErrorString([[maybe_unused]] cudaError_t error) -> const char*
{
  return "mock cuda error";
}

namespace {
constexpr int64_t kJobId = 7;
constexpr int64_t kWorkerId = 3;
constexpr double kLatencyMs = 123.0;

const std::vector<int64_t> kShape2x2{2, 2};
const std::vector<int64_t> kShape2x3{2, 3};
const std::vector<int64_t> kShape1{1};

const std::vector<torch::Dtype> kTypesFloatInt{torch::kFloat32, torch::kInt64};

struct ExpectedJobInfo {
  int64_t job_id;
  int worker_id;
};

inline auto
JobStateMatches(
    const std::shared_ptr<starpu_server::InferenceJob>& job,
    const std::vector<torch::Tensor>& inputs,
    const std::vector<at::ScalarType>& types,
    const std::vector<torch::Tensor>& outputs,
    std::chrono::high_resolution_clock::time_point start,
    const ExpectedJobInfo& expected) -> ::testing::AssertionResult
{
  if (job->get_job_id() != expected.job_id) {
    return ::testing::AssertionFailure() << "job_id mismatch";
  }
  if (job->get_input_tensors().size() != inputs.size()) {
    return ::testing::AssertionFailure() << "input count mismatch";
  }
  if (!job->get_input_tensors()[0].equal(inputs[0])) {
    return ::testing::AssertionFailure() << "input[0] tensor mismatch";
  }
  if (job->get_input_types().size() != types.size()) {
    return ::testing::AssertionFailure() << "input type count mismatch";
  }
  if (job->get_input_types()[0] != types[0]) {
    return ::testing::AssertionFailure() << "input type[0] mismatch";
  }
  if (job->get_output_tensors().size() != outputs.size()) {
    return ::testing::AssertionFailure() << "output count mismatch";
  }
  if (!job->get_output_tensors()[0].equal(outputs[0])) {
    return ::testing::AssertionFailure() << "output[0] tensor mismatch";
  }
  if (job->get_start_time() != start) {
    return ::testing::AssertionFailure() << "start time mismatch";
  }
  const auto fixed = job->get_fixed_worker_id();
  if (!fixed.has_value()) {
    return ::testing::AssertionFailure() << "fixed worker not set";
  }
  if (fixed.value() != expected.worker_id) {
    return ::testing::AssertionFailure() << "fixed worker mismatch";
  }
  if (!job->has_on_complete()) {
    return ::testing::AssertionFailure() << "on_complete not set";
  }
  return ::testing::AssertionSuccess();
}

inline auto
CallbackResultsMatch(
    bool called, const std::vector<torch::Tensor>& tensors, double latency,
    const std::vector<torch::Tensor>& outputs,
    double expected_latency) -> ::testing::AssertionResult
{
  if (!called) {
    return ::testing::AssertionFailure() << "callback not called";
  }
  if (tensors.size() != outputs.size()) {
    return ::testing::AssertionFailure() << "callback tensors size mismatch";
  }
  if (!tensors[0].equal(outputs[0])) {
    return ::testing::AssertionFailure() << "callback tensor[0] mismatch";
  }
  if (latency != expected_latency) {
    return ::testing::AssertionFailure() << "latency mismatch";
  }
  return ::testing::AssertionSuccess();
}
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
  job->set_output_tensors(outputs);
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

  EXPECT_TRUE(JobStateMatches(
      job, inputs, types, outputs, start,
      ExpectedJobInfo{
          .job_id = kJobId, .worker_id = static_cast<int>(kWorkerId)}));

  job->get_on_complete()(job->get_output_tensors(), kLatencyMs);
  EXPECT_TRUE(CallbackResultsMatch(
      callback_called, cb_tensors, cb_latency, outputs, kLatencyMs));
}

TEST(InferenceRunnerUtils_Unit, GenerateInputsShapeAndType)
{
  const std::vector<std::vector<int64_t>> shapes{kShape2x3, kShape1};
  std::vector<starpu_server::TensorConfig> cfgs;
  cfgs.reserve(shapes.size());
  for (size_t i = 0; i < shapes.size(); ++i) {
    cfgs.push_back(
        {std::string("input") + std::to_string(i), shapes[i],
         kTypesFloatInt[i]});
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
  constexpr float kF1 = 1.0F;
  constexpr float kF2 = 2.0F;
  constexpr float kF35 = 3.5F;
  constexpr float kFNeg4 = -4.0F;
  constexpr float kF025 = 0.25F;
  constexpr size_t kCount5 = 5;
  auto tensor = torch::tensor({kF1, kF2, kF35, kFNeg4, kF025}, torch::kFloat32);
  std::vector<float> dst(kCount5, 0.0F);
  starpu_server::TensorBuilder::copy_output_to_buffer(
      tensor, dst.data(), tensor.numel(), tensor.scalar_type());
  ASSERT_EQ(dst.size(), kCount5);
  EXPECT_FLOAT_EQ(dst[0], kF1);
  EXPECT_FLOAT_EQ(dst[1], kF2);
  EXPECT_FLOAT_EQ(dst[2], kF35);
  EXPECT_FLOAT_EQ(dst[3], kFNeg4);
  EXPECT_FLOAT_EQ(dst[4], kF025);
}

TEST(InferenceRunner_Unit, LogsCudaSyncError)
{
  SetCudaSyncResult(cudaErrorUnknown);
  starpu_server::CaptureStream capture{std::cerr};
  EXPECT_EQ(
      capture.str(), starpu_server::expected_log_line(
                         starpu_server::ErrorLevel,
                         "cudaDeviceSynchronize failed: mock cuda error"));
  SetCudaSyncResult(cudaSuccess);
}
