#include <gtest/gtest.h>
#include <torch/script.h>

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>

#include "core/inference_runner.hpp"
#include "test_helpers.hpp"
#include "test_inference_runner.hpp"
#include "utils/exceptions.hpp"

TEST(InferenceRunnerHelpers_Robustesse, LoadModelAndReferenceOutputCorruptFile)
{
  namespace fs = std::filesystem;
  auto tmp_path = fs::temp_directory_path() / "corrupt_model.pt";
  {
    std::ofstream tmp_file{tmp_path};
    tmp_file << "invalid";
  }

  starpu_server::RuntimeConfig opts;
  opts.models.resize(1);
  opts.models[0].path = tmp_path.string();
  opts.models[0].inputs = {{"input0", {1}, at::kFloat}};

  starpu_server::CaptureStream capture{std::cerr};
  auto result = starpu_server::load_model_and_reference_output(opts);
  auto err = capture.str();

  EXPECT_EQ(result, std::nullopt);
  EXPECT_NE(
      err.find("Failed to load model or run reference inference"),
      std::string::npos);

  fs::remove(tmp_path);
}

TEST(InferenceRunnerHelpers_Robustesse, RunReferenceInferenceUnsupportedOutput)
{
  auto model = starpu_server::make_constant_model();
  std::vector<torch::Tensor> inputs{torch::ones({1})};
  EXPECT_THROW(
      starpu_server::run_reference_inference(model, inputs),
      starpu_server::UnsupportedModelOutputTypeException);
}

class LoadModelAndReferenceOutputError_Robustesse
    : public ::testing::TestWithParam<at::ScalarType> {};

TEST_P(LoadModelAndReferenceOutputError_Robustesse, MissingFile)
{
  starpu_server::RuntimeConfig opts;
  opts.models.resize(1);
  opts.models[0].path = "nonexistent_model.pt";
  opts.models[0].inputs = {{"input0", {1}, GetParam()}};
  starpu_server::CaptureStream capture{std::cerr};
  auto result = starpu_server::load_model_and_reference_output(opts);
  auto err = capture.str();
  EXPECT_EQ(result, std::nullopt);
  EXPECT_NE(
      err.find("Failed to load model or run reference inference"),
      std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
    InferenceRunnerHelpers, LoadModelAndReferenceOutputError_Robustesse,
    ::testing::Values(torch::kFloat32, at::kFloat));
