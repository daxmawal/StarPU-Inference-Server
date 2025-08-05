#include <gtest/gtest.h>
#include <torch/script.h>

#include <iostream>
#include <vector>

#include "../test_helpers.hpp"
#include "core/inference_runner.hpp"
#include "inference_runner_test_utils.hpp"
#include "utils/exceptions.hpp"

TEST(InferenceRunnerErrors, LoadModelAndReferenceOutputMissingFile)
{
  starpu_server::RuntimeConfig opts;
  opts.model_path = "nonexistent_model.pt";
  opts.input_shapes = {{1}};
  opts.input_types = {at::kFloat};
  starpu_server::CaptureStream capture{std::cerr};
  auto [cpu_model, gpu_models, refs] =
      starpu_server::load_model_and_reference_output(opts);
  auto err = capture.str();
  EXPECT_TRUE(gpu_models.empty());
  EXPECT_TRUE(refs.empty());
  EXPECT_NE(
      err.find("Failed to load model or run reference inference"),
      std::string::npos);
}