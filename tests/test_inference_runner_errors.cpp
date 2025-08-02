#include <gtest/gtest.h>
#include <torch/script.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "core/inference_runner.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"

using namespace starpu_server;

TEST(InferenceRunnerErrors, LoadModelAndReferenceOutputMissingFile)
{
  RuntimeConfig opts;
  opts.model_path = "nonexistent_model.pt";
  opts.input_shapes = {{1}};
  opts.input_types = {at::kFloat};

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

// Local copy of run_reference_inference from inference_runner.cpp
static auto
run_reference_inference(
    torch::jit::script::Module& model,
    const std::vector<torch::Tensor>& inputs) -> std::vector<torch::Tensor>
{
  std::vector<torch::Tensor> output_refs;
  const std::vector<torch::IValue> input_ivalues(inputs.begin(), inputs.end());

  if (const auto output = model.forward(input_ivalues); output.isTensor()) {
    output_refs.push_back(output.toTensor());
  } else if (output.isTuple()) {
    for (const auto& val : output.toTuple()->elements()) {
      if (val.isTensor()) {
        output_refs.push_back(val.toTensor());
      }
    }
  } else if (output.isTensorList()) {
    output_refs.insert(
        output_refs.end(), output.toTensorList().begin(),
        output.toTensorList().end());
  } else {
    log_error("Unsupported output type from model.");
    throw UnsupportedModelOutputTypeException("Unsupported model output type");
  }

  return output_refs;
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
      run_reference_inference(m, inputs), UnsupportedModelOutputTypeException);
}
