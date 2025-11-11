#include <gtest/gtest.h>

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

}}  // namespace starpu_server
