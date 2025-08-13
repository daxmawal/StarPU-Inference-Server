#pragma once

#include <torch/script.h>

#include <filesystem>
#include <tuple>
#include <vector>

#include "runtime_config.hpp"
#include "utils/exceptions.hpp"
#include "utils/input_generator.hpp"
#include "utils/logger.hpp"

namespace starpu_server {

inline auto
make_identity_model() -> torch::jit::script::Module
{
  torch::jit::script::Module module{"m"};
  module.define(R"JIT(
        def forward(self, x):
            return x
    )JIT");
  return module;
}

inline auto
make_mul_two_model() -> torch::jit::script::Module
{
  torch::jit::script::Module module{"m"};
  module.define(R"JIT(
        def forward(self, x):
            return x * 2
    )JIT");
  return module;
}

inline auto
make_tuple_model() -> torch::jit::script::Module
{
  torch::jit::script::Module module{"m"};
  module.define(R"JIT(
        def forward(self, x):
            return (x, x + 1)
    )JIT");
  return module;
}

inline auto
make_tensor_list_model() -> torch::jit::script::Module
{
  torch::jit::script::Module module{"m"};
  module.define(R"JIT(
        def forward(self, x):
            return [x, x + 1]
    )JIT");
  return module;
}

inline auto
make_constant_model() -> torch::jit::script::Module
{
  torch::jit::script::Module module{"m"};
  module.define(R"JIT(
        def forward(self, x):
            return 5
    )JIT");
  return module;
}

inline auto
save_mul_two_model(const std::filesystem::path& path) -> void
{
  auto module = make_mul_two_model();
  module.save(path.string());
}

inline auto
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

inline auto
generate_inputs(
    const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<at::ScalarType>& types) -> std::vector<torch::Tensor>
{
  return input_generator::generate_random_inputs(shapes, types);
}

auto load_model_and_reference_output(const RuntimeConfig& opts)
    -> std::tuple<
        torch::jit::script::Module, std::vector<torch::jit::script::Module>,
        std::vector<torch::Tensor>>;
}  // namespace starpu_server
