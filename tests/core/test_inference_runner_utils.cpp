#include <gtest/gtest.h>
#include <torch/script.h>

#include <filesystem>

#include "inference_runner_test_utils.hpp"

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
  create_test_module(file);

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
  auto expected = starpu_server::run_reference_inference(cpu_model, inputs);

  ASSERT_EQ(refs.size(), expected.size());
  ASSERT_EQ(refs.size(), 1u);
  EXPECT_TRUE(torch::allclose(refs[0], expected[0]));

  std::filesystem::remove(file);
}
