#include <gtest/gtest.h>
#include <torch/script.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "core/inference_runner.hpp"
#include "inference_runner_test_utils.hpp"

namespace {
constexpr int64_t kJobId = 7;
constexpr int64_t kWorkerId = 3;
constexpr int kDeviceId0 = 0;
constexpr double kLatencyMs = 123.0;

constexpr uint64_t kSeed0 = 0ULL;
constexpr uint64_t kSeed1 = 1ULL;
constexpr uint64_t kSeed2 = 2ULL;
constexpr uint64_t kSeed3 = 3ULL;
constexpr uint64_t kSeedRef = 42ULL;

const std::vector<int64_t> kShape2x2{2, 2};
const std::vector<int64_t> kShape2x3{2, 3};
const std::vector<int64_t> kShape1{1};
const std::vector<int64_t> kShape2{2};
const std::vector<int64_t> kShape4{4};

const std::vector<torch::Dtype> kTypesFloat{torch::kFloat32};
const std::vector<torch::Dtype> kTypesFloatInt{torch::kFloat32, torch::kInt64};

constexpr const char* kTinyModuleBase = "tiny_module";
constexpr const char* kTupleModuleBase = "tuple_module";
constexpr const char* kTensorListModuleBase = "tensor_list_module";
constexpr const char* kConstantModuleBase = "constant_module";
constexpr const char* kModelExt = ".pt";

inline std::filesystem::path
MakeTempModelPath(const char* base_name)
{
  const auto dir = std::filesystem::temp_directory_path();
  const auto ts =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return dir / (std::string(base_name) + "_" + std::to_string(ts) + kModelExt);
}

}  // namespace

TEST(InferenceRunner, MakeShutdownJob)
{
  auto job = starpu_server::InferenceJob::make_shutdown_job();
  ASSERT_NE(job, nullptr);
  EXPECT_TRUE(job->is_shutdown());
}

TEST(InferenceJob, SettersAndGettersAndCallback)
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

  job->set_on_complete(
      [&](std::vector<torch::Tensor> tensor, double latency_ms) {
        callback_called = true;
        cb_tensors = std::move(tensor);
        cb_latency = latency_ms;
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
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_TRUE(cb_tensors[0].equal(outputs[0]));
  EXPECT_DOUBLE_EQ(cb_latency, kLatencyMs);
}

TEST(InferenceRunnerUtils, GenerateInputsShapeAndType)
{
  const std::vector<std::vector<int64_t>> shapes{kShape2x3, kShape1};
  const std::vector<torch::Dtype> types{kTypesFloatInt};

  torch::manual_seed(kSeed0);
  auto tensors = starpu_server::generate_inputs(shapes, types);
  ASSERT_EQ(tensors.size(), 2U);
  EXPECT_EQ(
      tensors[0].sizes(), (torch::IntArrayRef{kShape2x3[0], kShape2x3[1]}));
  EXPECT_EQ(tensors[0].dtype(), torch::kFloat32);
  EXPECT_EQ(tensors[1].sizes(), (torch::IntArrayRef{kShape1[0]}));
  EXPECT_EQ(tensors[1].dtype(), torch::kInt64);
}

TEST(InferenceRunnerUtils, LoadModelAndReferenceOutputCPU)
{
  const auto file = MakeTempModelPath(kTinyModuleBase);
  starpu_server::save_mul_two_model(file);

  starpu_server::RuntimeConfig opts;
  opts.model_path = file.string();
  opts.input_shapes = {kShape4};
  opts.input_types = kTypesFloat;
  opts.device_ids = {kDeviceId0};
  opts.use_cuda = false;

  torch::manual_seed(kSeedRef);
  auto [cpu_model, gpu_models, refs] =
      starpu_server::load_model_and_reference_output(opts);
  EXPECT_TRUE(gpu_models.empty());

  torch::manual_seed(kSeedRef);
  auto inputs =
      starpu_server::generate_inputs(opts.input_shapes, opts.input_types);
  ASSERT_EQ(refs.size(), 1U);
  EXPECT_TRUE(torch::allclose(refs[0], inputs[0] * 2));

  std::filesystem::remove(file);
}

TEST(InferenceRunnerUtils, LoadModelAndReferenceOutputTuple)
{
  const auto file = MakeTempModelPath(kTupleModuleBase);
  auto model = starpu_server::make_tuple_model();
  model.save(file.string());

  starpu_server::RuntimeConfig opts;
  opts.model_path = file.string();
  opts.input_shapes = {kShape2};
  opts.input_types = kTypesFloat;
  opts.device_ids = {kDeviceId0};
  opts.use_cuda = false;

  torch::manual_seed(kSeed1);
  auto [cpu_model, gpu_models, refs] =
      starpu_server::load_model_and_reference_output(opts);
  EXPECT_TRUE(gpu_models.empty());

  torch::manual_seed(kSeed1);
  auto inputs =
      starpu_server::generate_inputs(opts.input_shapes, opts.input_types);
  ASSERT_EQ(refs.size(), 2U);
  EXPECT_TRUE(torch::allclose(refs[0], inputs[0]));
  EXPECT_TRUE(torch::allclose(refs[1], inputs[0] + 1));

  std::filesystem::remove(file);
}

TEST(InferenceRunnerUtils, LoadModelAndReferenceOutputTensorList)
{
  const auto file = MakeTempModelPath(kTensorListModuleBase);
  auto model = starpu_server::make_tensor_list_model();
  model.save(file.string());

  starpu_server::RuntimeConfig opts;
  opts.model_path = file.string();
  opts.input_shapes = {kShape2};
  opts.input_types = kTypesFloat;
  opts.device_ids = {kDeviceId0};
  opts.use_cuda = false;

  torch::manual_seed(kSeed2);
  auto [cpu_model, gpu_models, refs] =
      starpu_server::load_model_and_reference_output(opts);
  EXPECT_TRUE(gpu_models.empty());

  torch::manual_seed(kSeed2);
  auto inputs =
      starpu_server::generate_inputs(opts.input_shapes, opts.input_types);
  ASSERT_EQ(refs.size(), 2U);
  EXPECT_TRUE(torch::allclose(refs[0], inputs[0]));
  EXPECT_TRUE(torch::allclose(refs[1], inputs[0] + 1));

  std::filesystem::remove(file);
}

TEST(InferenceRunnerUtils, LoadModelAndReferenceOutputUnsupported)
{
  const auto file = MakeTempModelPath(kConstantModuleBase);
  auto model = starpu_server::make_constant_model();
  model.save(file.string());

  starpu_server::RuntimeConfig opts;
  opts.model_path = file.string();
  opts.input_shapes = {kShape1};
  opts.input_types = kTypesFloat;
  opts.device_ids = {kDeviceId0};
  opts.use_cuda = false;

  torch::manual_seed(kSeed3);
  EXPECT_THROW(
      (void)starpu_server::load_model_and_reference_output(opts),
      starpu_server::UnsupportedModelOutputTypeException);

  std::filesystem::remove(file);
}
