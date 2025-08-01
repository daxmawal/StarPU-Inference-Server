#include <gtest/gtest.h>
#include <torch/script.h>

#include <filesystem>

#include "core/inference_runner.hpp"
#include "utils/input_generator.hpp"

using namespace starpu_server;
namespace fs = std::filesystem;

// Build a tiny TorchScript module for testing
static auto
create_test_module(const fs::path& path) -> void
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
  return input_generator::generate_random_inputs(shapes, types);
}

// Local copy of run_reference_inference from inference_runner.cpp
static auto
run_reference_inference(
    torch::jit::script::Module& model,
    const std::vector<torch::Tensor>& inputs) -> std::vector<torch::Tensor>
{
  std::vector<torch::Tensor> outputs;
  const std::vector<torch::IValue> ivals(inputs.begin(), inputs.end());
  auto output = model.forward(ivals);
  if (output.isTensor()) {
    outputs.push_back(output.toTensor());
  } else if (output.isTuple()) {
    for (const auto& v : output.toTuple()->elements()) {
      if (v.isTensor())
        outputs.push_back(v.toTensor());
    }
  } else if (output.isTensorList()) {
    outputs.insert(
        outputs.end(), output.toTensorList().begin(),
        output.toTensorList().end());
  }
  return outputs;
}

// Local copy of load_model_and_reference_output to avoid StarPU dependency
static auto
load_model_and_reference_output_local(const RuntimeConfig& opts)
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
  auto refs = run_reference_inference(model_cpu, inputs);
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
  fs::path file{"tiny_module.pt"};
  create_test_module(file);

  RuntimeConfig opts;
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
  auto expected = run_reference_inference(cpu_model, inputs);

  ASSERT_EQ(refs.size(), expected.size());
  ASSERT_EQ(refs.size(), 1u);
  EXPECT_TRUE(torch::allclose(refs[0], expected[0]));

  fs::remove(file);
}