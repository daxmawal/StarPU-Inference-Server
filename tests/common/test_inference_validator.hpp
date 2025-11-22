#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <utility>
#include <vector>

#include "core/inference_runner.hpp"

namespace starpu_server {

inline auto
make_result(
    std::vector<torch::Tensor> inputs, std::vector<torch::Tensor> outputs,
    int request_id, DeviceType device, int device_id = 0,
    int worker_id = 0) -> InferenceResult
{
  InferenceResult result;
  result.inputs = std::move(inputs);
  result.results = std::move(outputs);
  result.request_id = request_id;
  result.executed_on = device;
  result.device_id = device_id;
  result.worker_id = worker_id;
  return result;
}

}  // namespace starpu_server

class InferenceValidatorTest : public ::testing::Test {};

static auto
make_error_model() -> torch::jit::script::Module
{
  torch::jit::script::Module module{"m"};
  module.define(R"JIT(
      def forward(self, x):
          return torch.mm(x, x)
  )JIT");
  return module;
}

static auto
make_empty_tuple_model() -> torch::jit::script::Module
{
  torch::jit::script::Module module{"m"};
  module.define(R"JIT(
      def forward(self, x):
          return ()
  )JIT");
  return module;
}

static auto
make_tuple_non_tensor_model() -> torch::jit::script::Module
{
  torch::jit::script::Module module{"m"};
  module.define(R"JIT(
      def forward(self, x):
          return (x, 1)
  )JIT");
  return module;
}

static auto
make_tensor_list_model() -> torch::jit::script::Module
{
  torch::jit::script::Module module{"m"};
  module.define(R"JIT(
      def forward(self, x):
          return [x, x + 1]
  )JIT");
  return module;
}

static auto
make_string_model() -> torch::jit::script::Module
{
  torch::jit::script::Module module{"m"};
  module.define(R"JIT(
      def forward(self, x):
          return "hello"
  )JIT");
  return module;
}

static auto
make_shape_error_model() -> torch::jit::script::Module
{
  torch::jit::script::Module module{"m"};
  module.define(R"JIT(
      def forward(self, x):
          return x.view(-1, 0)
  )JIT");
  return module;
}
