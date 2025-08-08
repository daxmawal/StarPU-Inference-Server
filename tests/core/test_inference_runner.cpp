#include <gtest/gtest.h>
#include <torch/script.h>

#include <chrono>
#include <filesystem>
#include <vector>

#include "core/inference_runner.hpp"
#include "inference_runner_test_utils.hpp"

TEST(InferenceRunner, MakeShutdownJob)
{
  auto job = starpu_server::InferenceJob::make_shutdown_job();
  ASSERT_NE(job, nullptr);
  EXPECT_TRUE(job->is_shutdown());
}

TEST(InferenceJob, SettersAndGettersAndCallback)
{
  std::vector<torch::Tensor> inputs{torch::ones({2, 2})};
  std::vector<at::ScalarType> types{at::kFloat};
  std::vector<torch::Tensor> outputs{torch::zeros({2, 2})};
  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_job_id(7);
  job->set_input_tensors(inputs);
  job->set_input_types(types);
  job->set_outputs_tensors(outputs);
  job->set_fixed_worker_id(3);
  auto start = std::chrono::high_resolution_clock::now();
  job->set_start_time(start);
  bool callback_called = false;
  std::vector<torch::Tensor> cb_tensors;
  double cb_latency = 0.0;
  job->set_on_complete([&](std::vector<torch::Tensor> t, double latency) {
    callback_called = true;
    cb_tensors = std::move(t);
    cb_latency = latency;
  });
  EXPECT_EQ(job->get_job_id(), 7);
  ASSERT_EQ(job->get_input_tensors().size(), 1);
  EXPECT_TRUE(job->get_input_tensors()[0].equal(inputs[0]));
  ASSERT_EQ(job->get_input_types().size(), 1);
  EXPECT_EQ(job->get_input_types()[0], at::kFloat);
  ASSERT_EQ(job->get_output_tensors().size(), 1);
  EXPECT_TRUE(job->get_output_tensors()[0].equal(outputs[0]));
  EXPECT_EQ(job->get_start_time(), start);
  ASSERT_TRUE(job->get_fixed_worker_id().has_value());
  EXPECT_EQ(job->get_fixed_worker_id().value(), 3);
  ASSERT_TRUE(job->has_on_complete());
  job->get_on_complete()(job->get_output_tensors(), 123.0);
  EXPECT_TRUE(callback_called);
  ASSERT_EQ(cb_tensors.size(), 1);
  EXPECT_TRUE(cb_tensors[0].equal(outputs[0]));
  EXPECT_DOUBLE_EQ(cb_latency, 123.0);
}

TEST(InferenceRunnerUtils, GenerateInputsShapeAndType)
{
  std::vector<std::vector<int64_t>> shapes{{2, 3}, {1}};
  std::vector<torch::Dtype> types{torch::kFloat32, torch::kInt64};
  torch::manual_seed(0);
  auto tensors = starpu_server::generate_inputs(shapes, types);
  ASSERT_EQ(tensors.size(), 2u);
  EXPECT_EQ(tensors[0].sizes(), (torch::IntArrayRef{2, 3}));
  EXPECT_EQ(tensors[0].dtype(), torch::kFloat32);
  EXPECT_EQ(tensors[1].sizes(), (torch::IntArrayRef{1}));
  EXPECT_EQ(tensors[1].dtype(), torch::kInt64);
}

TEST(InferenceRunnerUtils, LoadModelAndReferenceOutputCPU)
{
  std::filesystem::path file{"tiny_module.pt"};
  starpu_server::save_mul_two_model(file);
  starpu_server::RuntimeConfig opts;
  opts.model_path = file.string();
  opts.input_shapes = {{4}};
  opts.input_types = {torch::kFloat32};
  opts.device_ids = {0};
  opts.use_cuda = false;
  torch::manual_seed(42);
  auto [cpu_model, gpu_models, refs] =
      starpu_server::load_model_and_reference_output(opts);
  EXPECT_TRUE(gpu_models.empty());
  torch::manual_seed(42);
  auto inputs =
      starpu_server::generate_inputs(opts.input_shapes, opts.input_types);
  ASSERT_EQ(refs.size(), 1u);
  EXPECT_TRUE(torch::allclose(refs[0], inputs[0] * 2));
  std::filesystem::remove(file);
}

TEST(InferenceRunnerUtils, LoadModelAndReferenceOutputTuple)
{
  std::filesystem::path file{"tuple_module.pt"};
  auto m = starpu_server::make_tuple_model();
  m.save(file.string());
  starpu_server::RuntimeConfig opts;
  opts.model_path = file.string();
  opts.input_shapes = {{2}};
  opts.input_types = {torch::kFloat32};
  opts.device_ids = {0};
  opts.use_cuda = false;
  torch::manual_seed(1);
  auto [cpu_model, gpu_models, refs] =
      starpu_server::load_model_and_reference_output(opts);
  EXPECT_TRUE(gpu_models.empty());
  torch::manual_seed(1);
  auto inputs =
      starpu_server::generate_inputs(opts.input_shapes, opts.input_types);
  ASSERT_EQ(refs.size(), 2u);
  EXPECT_TRUE(torch::allclose(refs[0], inputs[0]));
  EXPECT_TRUE(torch::allclose(refs[1], inputs[0] + 1));
  std::filesystem::remove(file);
}

TEST(InferenceRunnerUtils, LoadModelAndReferenceOutputTensorList)
{
  std::filesystem::path file{"tensor_list_module.pt"};
  auto m = starpu_server::make_tensor_list_model();
  m.save(file.string());
  starpu_server::RuntimeConfig opts;
  opts.model_path = file.string();
  opts.input_shapes = {{2}};
  opts.input_types = {torch::kFloat32};
  opts.device_ids = {0};
  opts.use_cuda = false;
  torch::manual_seed(2);
  auto [cpu_model, gpu_models, refs] =
      starpu_server::load_model_and_reference_output(opts);
  EXPECT_TRUE(gpu_models.empty());
  torch::manual_seed(2);
  auto inputs =
      starpu_server::generate_inputs(opts.input_shapes, opts.input_types);
  ASSERT_EQ(refs.size(), 2u);
  EXPECT_TRUE(torch::allclose(refs[0], inputs[0]));
  EXPECT_TRUE(torch::allclose(refs[1], inputs[0] + 1));
  std::filesystem::remove(file);
}

TEST(InferenceRunnerUtils, LoadModelAndReferenceOutputUnsupported)
{
  std::filesystem::path file{"constant_module.pt"};
  auto m = starpu_server::make_constant_model();
  m.save(file.string());
  starpu_server::RuntimeConfig opts;
  opts.model_path = file.string();
  opts.input_shapes = {{1}};
  opts.input_types = {torch::kFloat32};
  opts.device_ids = {0};
  opts.use_cuda = false;
  torch::manual_seed(3);
  EXPECT_THROW(
      starpu_server::load_model_and_reference_output(opts),
      starpu_server::UnsupportedModelOutputTypeException);
  std::filesystem::remove(file);
}
