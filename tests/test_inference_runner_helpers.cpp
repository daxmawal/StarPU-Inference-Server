#include <gtest/gtest.h>
#include <torch/script.h>

#include <format>
#include <sstream>
#include <vector>

#include "core/inference_runner.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"

using namespace starpu_server;

// -----------------------------------------------------------------------------
// Local copy of run_reference_inference from inference_runner.cpp
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Stubbed validate_inference_result and process_results for unit testing
// -----------------------------------------------------------------------------
static bool
validate_inference_result_stub(
    const InferenceResult& result, torch::jit::script::Module& jit_model,
    VerbosityLevel verbosity)
{
  const std::vector<torch::IValue> ivals(
      result.inputs.begin(), result.inputs.end());
  auto out = jit_model.forward(ivals).toTensor();
  bool ok = torch::allclose(out, result.results[0]);
  if (ok) {
    log_info(
        verbosity, std::format(
                       "[Validator] Job {} passed on {}", result.job_id,
                       to_string(result.executed_on)));
  }
  return ok;
}

static void
process_results_local(
    const std::vector<InferenceResult>& results,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    VerbosityLevel verbosity)
{
  for (const auto& result : results) {
    if (!result.results[0].defined()) {
      log_error(std::format("[Client] Job {} failed.", result.job_id));
      continue;
    }

    torch::jit::script::Module* cpu_model = &model_cpu;
    if (result.executed_on == DeviceType::CUDA) {
      const auto device_id = static_cast<size_t>(result.device_id);
      if (device_id < models_gpu.size()) {
        cpu_model = &models_gpu[device_id];
      }
    }

    validate_inference_result_stub(result, *cpu_model, verbosity);
  }
}

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
  auto outputs = run_reference_inference(m, inputs);

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
  auto outputs = run_reference_inference(m, inputs);

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
  auto outputs = run_reference_inference(m, inputs);

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
      run_reference_inference(m, inputs), UnsupportedModelOutputTypeException);
}

// -----------------------------------------------------------------------------
// load_model_and_reference_output error path
// -----------------------------------------------------------------------------

TEST(InferenceRunnerHelpers, LoadModelAndReferenceOutputError)
{
  RuntimeConfig opts;
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

// -----------------------------------------------------------------------------
// process_results tests
// -----------------------------------------------------------------------------

TEST(InferenceRunnerHelpers, ProcessResultsLogsErrorOnUndefined)
{
  torch::jit::script::Module model_cpu("m");
  model_cpu.define(R"JIT(def forward(self, x): return x)JIT");
  std::vector<torch::jit::script::Module> models_gpu;

  InferenceResult result;
  result.job_id = 1;
  result.results = {torch::Tensor{}};  // undefined tensor

  std::ostringstream err;
  auto* old_cerr = std::cerr.rdbuf(err.rdbuf());
  process_results_local(
      {result}, model_cpu, models_gpu, VerbosityLevel::Silent);
  std::cerr.rdbuf(old_cerr);

  EXPECT_NE(err.str().find("Job 1 failed"), std::string::npos);
}

TEST(InferenceRunnerHelpers, ProcessResultsUsesCorrectCudaModel)
{
  torch::jit::script::Module model_cpu("cpu");
  model_cpu.define(R"JIT(def forward(self, x): return x + 1)JIT");

  torch::jit::script::Module model_gpu0 = model_cpu.clone();
  torch::jit::script::Module model_gpu1("gpu1");
  model_gpu1.define(R"JIT(def forward(self, x): return x + 2)JIT");
  std::vector<torch::jit::script::Module> models_gpu{model_gpu0, model_gpu1};

  auto input = torch::ones({1});
  InferenceResult result;
  result.job_id = 2;
  result.inputs = {input};
  result.results = {input + 2};
  result.executed_on = DeviceType::CUDA;
  result.device_id = 1;

  std::ostringstream out;
  auto* old_cout = std::cout.rdbuf(out.rdbuf());
  process_results_local({result}, model_cpu, models_gpu, VerbosityLevel::Info);
  std::cout.rdbuf(old_cout);

  EXPECT_NE(out.str().find("Job 2 passed on CUDA"), std::string::npos);
}
