#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "../../../src/core/inference_runner.cpp"
#include "test_helpers.hpp"
#include "utils/runtime_config.hpp"

namespace starpu_server { namespace {

auto
make_valid_model_config() -> ModelConfig
{
  ModelConfig model;
  model.name = "test-model";
  model.outputs = {
      TensorConfig{
          .name = "output0",
          .dims = {2, 3},
          .type = at::kFloat,
      },
      TensorConfig{
          .name = "output1",
          .dims = {1, 4},
          .type = at::kDouble,
      }};
  return model;
}

auto
make_runtime_config_for_model(const std::filesystem::path& path)
    -> RuntimeConfig
{
  RuntimeConfig opts;
  opts.models.resize(1);
  opts.models[0].path = path.string();
  opts.models[0].inputs = {TensorConfig{
      .name = "input0",
      .dims = {1},
      .type = at::kFloat,
  }};
  opts.devices.use_cuda = false;
  return opts;
}

TEST(SynthesizeOutputsFromConfig, ReturnsNulloptWhenNoModels)
{
  RuntimeConfig opts;
  auto outputs = synthesize_outputs_from_config(opts);
  EXPECT_FALSE(outputs.has_value());
}

TEST(SynthesizeOutputsFromConfig, ReturnsNulloptWhenOutputsMissing)
{
  RuntimeConfig opts;
  ModelConfig model;
  opts.models = {model};

  auto outputs = synthesize_outputs_from_config(opts);
  EXPECT_FALSE(outputs.has_value());
}

TEST(SynthesizeOutputsFromConfig, ReturnsNulloptWhenTypeMissing)
{
  RuntimeConfig opts;
  auto model = make_valid_model_config();
  model.outputs[0].type = at::ScalarType::Undefined;
  opts.models = {model};

  CaptureStream capture{std::cerr};
  auto outputs = synthesize_outputs_from_config(opts);

  EXPECT_FALSE(outputs.has_value());
  EXPECT_NE(capture.str().find("missing a valid data_type"), std::string::npos);
}

TEST(SynthesizeOutputsFromConfig, ReturnsNulloptWhenDimsMissing)
{
  RuntimeConfig opts;
  auto model = make_valid_model_config();
  model.outputs[0].dims.clear();
  opts.models = {model};

  CaptureStream capture{std::cerr};
  auto outputs = synthesize_outputs_from_config(opts);

  EXPECT_FALSE(outputs.has_value());
  EXPECT_NE(capture.str().find("missing dims"), std::string::npos);
}

TEST(SynthesizeOutputsFromConfig, ReturnsNulloptWhenDimNonPositive)
{
  RuntimeConfig opts;
  auto model = make_valid_model_config();
  model.outputs[0].dims[0] = 0;
  opts.models = {model};

  CaptureStream capture{std::cerr};
  auto outputs = synthesize_outputs_from_config(opts);

  EXPECT_FALSE(outputs.has_value());
  EXPECT_NE(capture.str().find("non-positive dimension"), std::string::npos);
}

TEST(SynthesizeOutputsFromConfig, CreatesOutputsWhenConfigValid)
{
  RuntimeConfig opts;
  opts.models = {make_valid_model_config()};

  auto outputs = synthesize_outputs_from_config(opts);

  ASSERT_TRUE(outputs.has_value());
  ASSERT_EQ(outputs->size(), 2U);
  EXPECT_TRUE(outputs->at(0).sizes().vec() == opts.models[0].outputs[0].dims);
  EXPECT_EQ(outputs->at(0).dtype(), torch::kFloat32);
  EXPECT_TRUE(outputs->at(1).sizes().vec() == opts.models[0].outputs[1].dims);
  EXPECT_EQ(outputs->at(1).dtype(), torch::kFloat64);
}

TEST(LoadModelAndReferenceOutput, LogsFallbackWhenSyntheticMissing)
{
  TemporaryModelFile model_file{"load_model_missing", make_add_one_model()};
  RuntimeConfig opts = make_runtime_config_for_model(model_file.path());
  opts.validation.validate_results = false;
  opts.verbosity = VerbosityLevel::Debug;
  opts.models[0].outputs = {TensorConfig{
      .name = "bad_output",
      .dims = {},
      .type = at::kFloat,
  }};

  CaptureStream capture{std::cout};
  const auto result = load_model_and_reference_output(opts);

  EXPECT_TRUE(result.has_value());
  EXPECT_NE(
      capture.str().find(
          "Validation disabled but missing usable output schema"),
      std::string::npos);
}

TEST(LoadModelAndReferenceOutput, LogsWhenUsingSyntheticOutputs)
{
  TemporaryModelFile model_file{"load_model_synthetic", make_add_one_model()};
  RuntimeConfig opts = make_runtime_config_for_model(model_file.path());
  opts.validation.validate_results = false;
  opts.verbosity = VerbosityLevel::Debug;
  opts.models[0].outputs = {TensorConfig{
      .name = "output0",
      .dims = {1},
      .type = at::kFloat,
  }};

  CaptureStream capture{std::cout};
  const auto result = load_model_and_reference_output(opts);

  ASSERT_TRUE(result.has_value());
  EXPECT_NE(
      capture.str().find(
          "Validation disabled; using configured output schema instead"),
      std::string::npos);
  const auto& outputs = std::get<2>(*result);
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_TRUE(outputs[0].defined());
  EXPECT_EQ(outputs[0].sizes().vec(), opts.models[0].outputs[0].dims);
}

TEST(InferenceRunner, LoadModelLoadsTorchScriptModule)
{
  TemporaryModelFile model_file{"load_model_basic", make_add_one_model()};

  auto module = load_model(model_file.path().string());

  const auto input = torch::ones({1}, torch::TensorOptions().dtype(at::kFloat));
  const auto output = module.forward({input}).toTensor();
  EXPECT_TRUE(output.allclose(input + 1));
}

TEST(InferenceRunner, CloneModelToGpusReturnsEmptyWhenNoDeviceIds)
{
  auto cpu_model = make_add_one_model();
  const std::vector<int> device_ids;

  const auto gpu_models = clone_model_to_gpus(cpu_model, device_ids);

  EXPECT_TRUE(gpu_models.empty());
}

}}  // namespace starpu_server
