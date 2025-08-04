#include <gtest/gtest.h>
#include <torch/script.h>

#include <filesystem>

#include "core/inference_runner.hpp"
#include "inference_runner_test_utils.hpp"
#include "utils/input_generator.hpp"

// Build a tiny TorchScript module for testing
static auto
create_test_module(const std::filesystem::path& path) -> void
{
  torch::jit::Module m("m");
  m.define(R"JIT(
        def forward(self, x):
            return x * 2
    )JIT");
  m.save(path.string());
}

// Local copy of generate_inputs from inference_runner.cpp
static auto
generate_inputs(
    const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<torch::Dtype>& types) -> std::vector<torch::Tensor>
{
  return starpu_server::input_generator::generate_random_inputs(shapes, types);
}

// Local copy of load_model_and_reference_output to avoid StarPU dependency
static auto
load_model_and_reference_output_local(const starpu_server::RuntimeConfig& opts)
    -> std::tuple<
        torch::jit::script::Module, std::vector<torch::jit::script::Module>,
        std::vector<torch::Tensor>>
{
  auto model_cpu = torch::jit::load(opts.model_path);
  std::vector<torch::jit::script::Module> models_gpu;
  if (opts.use_cuda) {
    models_gpu.reserve(opts.device_ids.size());
    for (int id : opts.device_ids) {
      auto clone = model_cpu.clone();
      clone.to(torch::Device(torch::kCUDA, id));
      models_gpu.emplace_back(std::move(clone));
    }
  }
  auto inputs = generate_inputs(opts.input_shapes, opts.input_types);
  auto refs = starpu_server::run_reference_inference(model_cpu, inputs);
  return {model_cpu, models_gpu, refs};
}

TEST(InferenceRunnerUtils, GenerateInputsShapeAndType)
{
  std::vector<std::vector<int64_t>> shapes{{2, 3}, {1}};
  std::vector<torch::Dtype> types{torch::kFloat32, torch::kInt64};

  torch::manual_seed(0);
  auto tensors = generate_inputs(shapes, types);

  ASSERT_EQ(tensors.size(), 2u);
  EXPECT_EQ(tensors[0].sizes(), (torch::IntArrayRef{2, 3}));
  EXPECT_EQ(tensors[0].dtype(), torch::kFloat32);
  EXPECT_EQ(tensors[1].sizes(), (torch::IntArrayRef{1}));
  EXPECT_EQ(tensors[1].dtype(), torch::kInt64);
}

TEST(InferenceRunnerUtils, LoadModelAndReferenceOutputCPU)
{
  std::filesystem::path file{"tiny_module.pt"};
  create_test_module(file);

  starpu_server::RuntimeConfig opts;
  opts.model_path = file.string();
  opts.input_shapes = {{4}};
  opts.input_types = {torch::kFloat32};
  opts.device_ids = {0};
  opts.use_cuda = false;

  torch::manual_seed(42);
  auto [cpu_model, gpu_models, refs] =
      load_model_and_reference_output_local(opts);

  EXPECT_TRUE(gpu_models.empty());

  torch::manual_seed(42);
  auto inputs = generate_inputs(opts.input_shapes, opts.input_types);
  auto expected = starpu_server::run_reference_inference(cpu_model, inputs);

  ASSERT_EQ(refs.size(), expected.size());
  ASSERT_EQ(refs.size(), 1u);
  EXPECT_TRUE(torch::allclose(refs[0], expected[0]));

  std::filesystem::remove(file);
}
