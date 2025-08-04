#include <gtest/gtest.h>
#include <torch/script.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "core/inference_runner.hpp"
#include "inference_runner_test_utils.hpp"
#include "utils/exceptions.hpp"

TEST(InferenceRunnerErrors, LoadModelAndReferenceOutputMissingFile)
{
  starpu_server::RuntimeConfig opts;
  opts.model_path = "nonexistent_model.pt";
  opts.input_shapes = {{1}};
  opts.input_types = {at::kFloat};

  std::ostringstream oss;
  auto* old_cerr = std::cerr.rdbuf(oss.rdbuf());

  auto [cpu_model, gpu_models, refs] =
      starpu_server::load_model_and_reference_output(opts);

  std::cerr.rdbuf(old_cerr);

  EXPECT_TRUE(gpu_models.empty());
  EXPECT_TRUE(refs.empty());
  EXPECT_NE(
      oss.str().find("Failed to load model or run reference inference"),
      std::string::npos);
}

TEST(InferenceRunnerErrors, RunReferenceInferenceUnsupportedOutput)
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
