#include <gtest/gtest.h>
#include <torch/script.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "core/inference_runner.hpp"
#include "test_inference_runner.hpp"


namespace {
const std::vector<int64_t> kShape1{1};
const std::vector<torch::Dtype> kTypesFloat{torch::kFloat32};

inline std::filesystem::path
MakeTempModelPath(const char* base)
{
  const auto dir = std::filesystem::temp_directory_path();
  const auto ts =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return dir / (std::string(base) + "_" + std::to_string(ts) + ".pt");
}
}  // namespace

TEST(InferenceRunner_Robustesse, LoadModelAndReferenceOutputUnsupported)
{
  const auto file = MakeTempModelPath("constant_module");
  auto model = starpu_server::make_constant_model();
  model.save(file.string());

  starpu_server::RuntimeConfig opts;
  opts.model_path = file.string();
  opts.input_shapes = {kShape1};
  opts.input_types = kTypesFloat;
  opts.device_ids = {0};
  opts.use_cuda = false;

  torch::manual_seed(3);
  EXPECT_THROW(
      (void)starpu_server::load_model_and_reference_output(opts),
      starpu_server::UnsupportedModelOutputTypeException);

  std::filesystem::remove(file);
}
