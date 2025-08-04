#include <gtest/gtest.h>
#include <torch/script.h>

#include <format>
#include <sstream>
#include <vector>

#include "core/inference_runner.hpp"
#include "inference_runner_test_utils.hpp"
#include "utils/exceptions.hpp"

// -----------------------------------------------------------------------------
// run_reference_inference tests
// -----------------------------------------------------------------------------

TEST(InferenceRunnerHelpers, RunReferenceInferenceTensor)
{
  torch::jit::script::Module m("m");
  m.define(R"JIT(
        def forward(self, x):
            return x * 2
    )JIT");

  std::vector<torch::Tensor> inputs{torch::ones({2})};
  auto outputs = starpu_server::run_reference_inference(m, inputs);

  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_TRUE(torch::allclose(outputs[0], inputs[0] * 2));
}

TEST(InferenceRunnerHelpers, RunReferenceInferenceTuple)
{
  torch::jit::script::Module m("m");
  m.define(R"JIT(
        def forward(self, x):
            return (x, x + 1)
    )JIT");

  std::vector<torch::Tensor> inputs{torch::ones({2})};
  auto outputs = starpu_server::run_reference_inference(m, inputs);

  ASSERT_EQ(outputs.size(), 2u);
  EXPECT_TRUE(torch::allclose(outputs[0], inputs[0]));
  EXPECT_TRUE(torch::allclose(outputs[1], inputs[0] + 1));
}

TEST(InferenceRunnerHelpers, RunReferenceInferenceTensorList)
{
  torch::jit::script::Module m("m");
  m.define(R"JIT(
        def forward(self, x):
            return [x, x + 1]
    )JIT");

  std::vector<torch::Tensor> inputs{torch::ones({2})};
  auto outputs = starpu_server::run_reference_inference(m, inputs);

  ASSERT_EQ(outputs.size(), 2u);
  EXPECT_TRUE(torch::allclose(outputs[0], inputs[0]));
  EXPECT_TRUE(torch::allclose(outputs[1], inputs[0] + 1));
}

TEST(InferenceRunnerHelpers, RunReferenceInferenceUnsupported)
{
  torch::jit::script::Module m("m");
  m.define(R"JIT(
        def forward(self, x):
            return 5
    )JIT");

  std::vector<torch::Tensor> inputs{torch::ones({1})};

  EXPECT_THROW(
      starpu_server::run_reference_inference(m, inputs),
      starpu_server::UnsupportedModelOutputTypeException);
}

// -----------------------------------------------------------------------------
// load_model_and_reference_output error path
// -----------------------------------------------------------------------------

TEST(InferenceRunnerHelpers, LoadModelAndReferenceOutputError)
{
  starpu_server::RuntimeConfig opts;
  opts.model_path = "nonexistent_model.pt";
  opts.input_shapes = {{1}};
  opts.input_types = {torch::kFloat32};

  std::ostringstream oss;
  auto* old_cerr = std::cerr.rdbuf(oss.rdbuf());
  auto [cpu_model, gpu_models, refs] = load_model_and_reference_output(opts);
  std::cerr.rdbuf(old_cerr);

  EXPECT_TRUE(gpu_models.empty());
  EXPECT_TRUE(refs.empty());
  EXPECT_NE(
      oss.str().find("Failed to load model or run reference inference"),
      std::string::npos);
}
