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
        }()),
    [](const ::testing::TestParamInfo<ExtractTensorsParam>& info) {
      switch (info.index) {
        case 0:
          return std::string{"SingleTensor"};
        case 1:
          return std::string{"TupleOfTensors"};
        case 2:
          return std::string{"TensorList"};
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
