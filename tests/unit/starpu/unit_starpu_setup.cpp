#include <array>
#include <memory>
#include <string_view>

#include "test_starpu_setup.hpp"

TEST_P(StarPUSetupExtractTensorsTest, Extract)
{
  const auto& param = GetParam();
  auto outputs = starpu_server::extract_tensors_from_output(param.input);
  ASSERT_EQ(outputs.size(), param.expected.size());
  for (size_t i = 0; i < param.expected.size(); ++i) {
    EXPECT_TRUE(outputs[i].equal(param.expected[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(
    StarPUSetupExtractTensorsTestCases, StarPUSetupExtractTensorsTest,
    ::testing::Values(
        []() {
          at::Tensor tensor = torch::arange(6).view({2, 3});
          return ExtractTensorsParam{c10::IValue(tensor), {tensor}};
        }(),
        []() {
          at::Tensor tensor_1 = torch::ones({2, 2});
          at::Tensor tensor_2 = torch::zeros({1, 3});
          auto tuple = c10::ivalue::Tuple::create({tensor_1, tensor_2});
          return ExtractTensorsParam{c10::IValue(tuple), {tensor_1, tensor_2}};
        }(),
        []() {
          at::Tensor tensor_1 = torch::rand({2});
          at::Tensor tensor_2 = torch::rand({3});
          c10::List<at::Tensor> list;
          list.push_back(tensor_1);
          list.push_back(tensor_2);
          return ExtractTensorsParam{c10::IValue(list), {tensor_1, tensor_2}};
        }(),
        []() {
          at::Tensor tensor_1 = torch::rand({2});
          at::Tensor tensor_2 = torch::rand({3});
          at::Tensor tensor_3 = torch::rand({4});
          c10::impl::GenericList inner(c10::AnyType::get());
          inner.push_back(tensor_2);
          inner.push_back(tensor_3);
          c10::impl::GenericList outer(c10::AnyType::get());
          outer.push_back(tensor_1);
          outer.push_back(c10::IValue(inner));
          return ExtractTensorsParam{
              c10::IValue(outer), {tensor_1, tensor_2, tensor_3}};
        }(),
        []() {
          at::Tensor tensor_1 = torch::rand({2});
          at::Tensor tensor_2 = torch::rand({3});
          c10::Dict<std::string, at::Tensor> dict;
          dict.insert("first", tensor_1);
          dict.insert("second", tensor_2);
          return ExtractTensorsParam{c10::IValue(dict), {tensor_1, tensor_2}};
        }(),
        []() {
          at::Tensor tensor_1 = torch::rand({2});
          at::Tensor tensor_2 = torch::rand({3});
          at::Tensor tensor_3 = torch::rand({4});
          c10::impl::GenericDict inner(
              c10::StringType::get(), c10::AnyType::get());
          inner.insert(c10::IValue("second"), c10::IValue(tensor_2));
          inner.insert(c10::IValue("third"), c10::IValue(tensor_3));
          c10::impl::GenericDict outer(
              c10::StringType::get(), c10::AnyType::get());
          outer.insert(c10::IValue("first"), c10::IValue(tensor_1));
          outer.insert(c10::IValue("nested"), c10::IValue(inner));
          return ExtractTensorsParam{
              c10::IValue(outer), {tensor_1, tensor_2, tensor_3}};
        }()),
    [](const ::testing::TestParamInfo<ExtractTensorsParam>& info) {
      switch (info.index) {
        case 0:
          return std::string{"SingleTensor"};
        case 1:
          return std::string{"TupleOfTensors"};
        case 2:
          return std::string{"TensorList"};
        case 3:
          return std::string{"NestedList"};
        case 4:
          return std::string{"Dict"};
        case 5:
          return std::string{"NestedDict"};
        default:
          return std::string{"Unknown"};
      }
    });

TEST(InferenceCodelet, FieldsAreInitialized)
{
  starpu_server::InferenceCodelet inf_cl;
  auto* codelet = inf_cl.get_codelet();
  EXPECT_EQ(codelet->nbuffers, STARPU_VARIABLE_NBUFFERS);
  EXPECT_NE(codelet->cpu_funcs[0], nullptr);
  EXPECT_NE(codelet->cuda_funcs[0], nullptr);
  EXPECT_EQ(codelet->cuda_flags[0], 1U);
  EXPECT_EQ(codelet->max_parallelism, INT_MAX);
}

TEST(InferenceCodelet, RunInferenceExceptionsAreWrapped)
{
  const StarpuRuntimeGuard guard;

  auto params = starpu_server::make_basic_params(1);
  params.limits.max_inputs = 0;
  params.num_inputs = 1;

  float dummy_input = 0.0F;
  float dummy_output = 0.0F;
  std::array<starpu_variable_interface, 2> raw_buffers{};
  raw_buffers[0] = starpu_server::make_variable_interface(&dummy_input);
  raw_buffers[1] = starpu_server::make_variable_interface(&dummy_output);
  std::array<void*, 2> buffers{&raw_buffers[0], &raw_buffers[1]};

  starpu_server::InferenceCodelet inf_cl;
  auto* cpu_func = inf_cl.get_codelet()->cpu_funcs[0];
  ASSERT_NE(cpu_func, nullptr);

  try {
    cpu_func(buffers.data(), &params);
    FAIL() << "Expected cpu_func to throw";
  }
  catch (const starpu_server::StarPUCodeletException& ex) {
    const std::string_view message(ex.what());
    EXPECT_NE(message.find("[ERROR] Codelet failure"), std::string::npos);
  }
}

TEST(InferenceCodelet, SelectGpuModuleThrowsWhenReplicaMissing)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP();
  }

  auto params = starpu_server::make_basic_params(1);
  const int device_id = 0;

  EXPECT_THROW(
      starpu_server::select_gpu_module(params, device_id),
      starpu_server::StarPUCodeletException);
}

TEST(InferenceCodelet, SelectGpuModuleReturnsMatchingReplica)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP();
  }

  auto params = starpu_server::make_basic_params(1);
  const int device_id = 0;

  auto module = std::make_unique<torch::jit::script::Module>("dummy");
  params.models.models_gpu.resize(1);
  params.models.models_gpu[0] = module.get();
  params.models.num_models_gpu = params.models.models_gpu.size();

  torch::jit::script::Module* selected =
      starpu_server::select_gpu_module(params, device_id);

  EXPECT_EQ(selected, module.get());
}
