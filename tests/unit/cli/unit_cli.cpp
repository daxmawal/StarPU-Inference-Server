#include <gtest/gtest.h>
#include <torch/torch.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "test_cli.hpp"
#include "utils/runtime_config.hpp"

TEST(ArgsParser_Unit, ParsesRequiredOptions)
{
  const auto& model = test_model_path();
  std::vector<const char*> args{"program", "--model",     model.c_str(),
                                "--shape", "1x3x224x224", "--types",
                                "float"};
  const auto opts = parse(args);
  ASSERT_TRUE(opts.valid);
  EXPECT_EQ(opts.models[0].path, model);
  ASSERT_EQ(opts.models[0].inputs.size(), 1U);
  EXPECT_EQ(
      opts.models[0].inputs[0].dims, (std::vector<int64_t>{1, 3, 224, 224}));
  EXPECT_EQ(opts.models[0].inputs[0].type, at::kFloat);
}

TEST(ArgsParser_Unit, ParsesConfigOption)
{
  const auto& model = test_model_path();
  const auto& config = test_config_path();
  std::vector<const char*> args{"program", "--model",  model.c_str(),
                                "--shape", "1x1",      "--types",
                                "float",   "--config", config.c_str()};
  const auto opts = parse(args);
  ASSERT_TRUE(opts.valid);
  EXPECT_EQ(opts.config_path, config);
}

TEST(ArgsParser_Unit, ParsesSlotsAlias)
{
  const auto& model = test_model_path();
  constexpr int kSlots = 4;
  constexpr const char* kSlotsArg = "4";
  std::vector<const char*> args{"program", "--model", model.c_str(),
                                "--shape", "1x1",     "--types",
                                "float",   "--slots", kSlotsArg};

  const auto opts = parse(args);
  ASSERT_TRUE(opts.valid);
  EXPECT_EQ(opts.batching.pool_size, kSlots);
}

TEST(ArgsParser_Unit, RejectsNonPositiveSlotsAlias)
{
  const auto& model = test_model_path();
  for (const char* value : {"0", "-1"}) {
    std::vector<const char*> args{"program", "--model", model.c_str(),
                                  "--shape", "1x1",     "--types",
                                  "float",   "--slots", value};

    const auto opts = parse(args);
    EXPECT_FALSE(opts.valid);
  }
}

TEST(ArgsParser_Unit, ParsesPoolSize)
{
  const auto& model = test_model_path();
  std::vector<const char*> args{"program", "--model",     model.c_str(),
                                "--shape", "1x1",         "--types",
                                "float",   "--pool-size", "4"};
  const auto opts = parse(args);
  ASSERT_TRUE(opts.valid);
  EXPECT_EQ(opts.batching.pool_size, 4);
}

TEST(ArgsParser_Unit, ParsesCpuGroupByNumaFlag)
{
  const auto& model = test_model_path();
  std::vector<const char*> args{
      "program", "--model", model.c_str(), "--shape",
      "1x1",     "--types", "float",       "--cpu-group-by-numa"};
  const auto opts = parse(args);
  ASSERT_TRUE(opts.valid);
  EXPECT_TRUE(opts.devices.group_cpu_by_numa);
}

TEST(ArgsParser_Unit, ParsesDisableCpuGroupByNumaFlag)
{
  const auto& model = test_model_path();
  std::vector<const char*> args{
      "program",
      "--model",
      model.c_str(),
      "--shape",
      "1x1",
      "--types",
      "float",
      "--cpu-group-by-numa",
      "--no-cpu-group-by-numa"};
  const auto opts = parse(args);
  ASSERT_TRUE(opts.valid);
  EXPECT_FALSE(opts.devices.group_cpu_by_numa);
}

TEST(ArgsParser_Unit, ParsesLegacyInputSlotsFlag)
{
  const auto& model = test_model_path();
  std::vector<const char*> args{"program", "--model",       model.c_str(),
                                "--shape", "1x1",           "--types",
                                "float",   "--input-slots", "5"};
  const auto opts = parse(args);
  ASSERT_TRUE(opts.valid);
  EXPECT_EQ(opts.batching.pool_size, 5);
}

TEST(ArgsParser_Unit, ParsesAllOptions)
{
  const auto& model = test_model_path();

  const int device_count =
      static_cast<int>(static_cast<unsigned char>(torch::cuda::device_count()));

  std::vector<const char*> args{
      "program",
      "--model",
      model.c_str(),
      "--shapes",
      "1x3x224x224,2x1",
      "--types",
      "float,int",
      "--request-number",
      "5",
      "--verbose",
      "3",
      "--delay",
      "42",
      "--scheduler",
      "lws",
      "--address",
      "127.0.0.1:1234",
      "--max-batch-size",
      "2",
      "--pool-size",
      "8",
      "--pregen-inputs",
      "7",
      "--warmup-pregen-inputs",
      "5",
      "--warmup-request_nb",
      "3",
      "--seed",
      "123",
      "--sync",
      "--no_cpu"};

  if (device_count >= 2) {
    args.push_back("--device-ids");
    args.push_back("0,1");
  } else if (device_count == 1) {
    args.push_back("--device-ids");
    args.push_back("0");
  }

  const auto opts = parse(args);
  ASSERT_TRUE(opts.valid);
  EXPECT_EQ(opts.scheduler, "lws");
  EXPECT_EQ(opts.models[0].path, model);
  EXPECT_EQ(opts.batching.request_nb, 5);
  EXPECT_EQ(opts.batching.delay_us, 42);
  EXPECT_EQ(opts.verbosity, starpu_server::VerbosityLevel::Debug);
  EXPECT_EQ(opts.server_address, "127.0.0.1:1234");
  EXPECT_EQ(opts.batching.max_batch_size, 2);
  EXPECT_EQ(opts.batching.pool_size, 8);
  EXPECT_EQ(opts.batching.pregen_inputs, 7U);
  EXPECT_EQ(opts.batching.warmup_pregen_inputs, 5U);
  EXPECT_EQ(opts.batching.warmup_request_nb, 3);
  ASSERT_TRUE(opts.seed.has_value());
  EXPECT_EQ(opts.seed, 123U);
  EXPECT_TRUE(opts.validation.validate_results);
  constexpr std::size_t expected_bytes = 32ULL * 1024ULL * 1024ULL;
  EXPECT_EQ(opts.batching.max_message_bytes, expected_bytes);
  EXPECT_TRUE(opts.batching.synchronous);
  EXPECT_FALSE(opts.devices.use_cpu);

  if (device_count >= 2) {
    EXPECT_TRUE(opts.devices.use_cuda);
    ASSERT_EQ(opts.devices.ids, (std::vector<int>{0, 1}));
  } else if (device_count == 1) {
    EXPECT_TRUE(opts.devices.use_cuda);
    ASSERT_EQ(opts.devices.ids, (std::vector<int>{0}));
  } else {
    EXPECT_FALSE(opts.devices.use_cuda);
    ASSERT_TRUE(opts.devices.ids.empty());
  }

  ASSERT_EQ(opts.models[0].inputs.size(), 2U);
  EXPECT_EQ(
      opts.models[0].inputs[0].dims, (std::vector<int64_t>{1, 3, 224, 224}));
  EXPECT_EQ(opts.models[0].inputs[1].dims, (std::vector<int64_t>{2, 1}));
  EXPECT_EQ(opts.models[0].inputs[0].type, at::kFloat);
  EXPECT_EQ(opts.models[0].inputs[1].type, at::kInt);
}

TEST(ArgsParser_Unit, VerboseLevels)
{
  using enum starpu_server::VerbosityLevel;
  const std::array<std::pair<const char*, starpu_server::VerbosityLevel>, 5>
      cases = {
          {{"0", Silent},
           {"1", Info},
           {"2", Stats},
           {"3", Debug},
           {"4", Trace}}};
  const auto& model = test_model_path();
  for (const auto& [lvl, expected] : cases) {
    std::vector<const char*> args{"program", "--model",   model.c_str(),
                                  "--shape", "1x3",       "--types",
                                  "float",   "--verbose", lvl};
    const auto opts = parse(args);
    ASSERT_TRUE(opts.valid);
    EXPECT_EQ(opts.verbosity, expected);
  }
}

TEST(ArgsParser_Unit, ParsesMixedCaseTypes)
{
  const auto& model = test_model_path();
  std::vector<const char*> args{"program", "--model", model.c_str(), "--shapes",
                                "1x1,1x1", "--types", "FlOat,InT"};
  const auto opts = parse(args);
  ASSERT_TRUE(opts.valid);
  ASSERT_EQ(opts.models[0].inputs.size(), 2U);
  EXPECT_EQ(opts.models[0].inputs[0].type, at::kFloat);
  EXPECT_EQ(opts.models[0].inputs[1].type, at::kInt);
}

TEST(ArgsParser_Unit, ParsesCombinedInputWithName)
{
  std::vector<const char*> args{
      "program", "--model", test_model_path().c_str(), "--input",
      "custom:1x3:int"};

  const auto opts = parse(args);

  ASSERT_TRUE(opts.valid);
  ASSERT_EQ(opts.models[0].inputs.size(), 1U);
  const auto& input = opts.models[0].inputs[0];
  EXPECT_EQ(input.name, "custom");
  EXPECT_EQ(input.dims, (std::vector<int64_t>{1, 3}));
  EXPECT_EQ(input.type, at::kInt);
}

TEST(ArgsParser_Unit, ParsesMultipleCombinedInputs)
{
  std::vector<const char*> args{
      "program", "--model", test_model_path().c_str(), "--input", "3x4:float",
      "--input", "5x6:int"};

  const auto opts = parse(args);

  ASSERT_TRUE(opts.valid);
  ASSERT_EQ(opts.models[0].inputs.size(), 2U);

  const auto& first = opts.models[0].inputs[0];
  EXPECT_EQ(first.name, "input0");
  EXPECT_EQ(first.dims, (std::vector<int64_t>{3, 4}));
  EXPECT_EQ(first.type, at::kFloat);

  const auto& second = opts.models[0].inputs[1];
  EXPECT_EQ(second.name, "input1");
  EXPECT_EQ(second.dims, (std::vector<int64_t>{5, 6}));
  EXPECT_EQ(second.type, at::kInt);
}

TEST(ArgsParser_Unit, RejectsInvalidCombinedInputs)
{
  const auto& model = test_model_path();
  expect_invalid({"program", "--model", model.c_str(), "--input"});
  expect_invalid({"program", "--model", model.c_str(), "--input", "1x3"});
  expect_invalid(
      {"program", "--model", model.c_str(), "--input", "1x3:unknown"});
  expect_invalid({"program", "--model", model.c_str(), "--input", "0x3:int"});
  expect_invalid({"program", "--model", model.c_str(), "--input", "ax3:int"});
}

TEST(ArgsParser_Unit, CombinedInputOverridesShapeAndType)
{
  std::vector<const char*> args{"program", "--model", test_model_path().c_str(),
                                "--shape", "1x3",     "--types",
                                "float",   "--input", "2x4:int"};

  const auto opts = parse(args);

  ASSERT_TRUE(opts.valid);
  ASSERT_EQ(opts.models[0].inputs.size(), 1U);
  const auto& input = opts.models[0].inputs[0];
  EXPECT_EQ(input.name, "input0");
  EXPECT_EQ(input.dims, (std::vector<int64_t>{2, 4}));
  EXPECT_EQ(input.type, at::kInt);
}

TEST(ArgsParser_Unit, MetricsPortBoundaryValues)
{
  for (const int port : {1, 65535}) {
    std::string port_str = std::to_string(port);
    std::vector<const char*> args = {
        "program", "--model", test_model_path().c_str(), "--shape",       "1x1",
        "--types", "float",   "--metrics-port",          port_str.c_str()};
    const auto opts = parse(args);
    ASSERT_TRUE(opts.valid);
    EXPECT_EQ(opts.metrics_port, port);
  }
}

TEST(ArgsParser_Unit, ParsesToleranceOptions)
{
  const auto& model = test_model_path();
  std::vector<const char*> args{"program", "--model", model.c_str(), "--shape",
                                "1x1",     "--types", "float",       "--rtol",
                                "1e-2",    "--atol",  "1e-4"};
  const auto opts = parse(args);
  ASSERT_TRUE(opts.valid);
  EXPECT_DOUBLE_EQ(opts.validation.rtol, 1e-2);
  EXPECT_DOUBLE_EQ(opts.validation.atol, 1e-4);
}

TEST(ArgsParser_Unit, DisableValidationFlag)
{
  const auto& model = test_model_path();
  std::vector<const char*> args{"program", "--model",      model.c_str(),
                                "--shape", "1x1",          "--types",
                                "float",   "--no-validate"};
  const auto opts = parse(args);
  ASSERT_TRUE(opts.valid);
  EXPECT_FALSE(opts.validation.validate_results);
}

TEST(ArgsParser_Unit, RejectsInvalidPoolSize)
{
  const auto& model = test_model_path();
  expect_invalid(
      {"program", "--model", model.c_str(), "--shape", "1x1", "--types",
       "float", "--pool-size", "0"});
  expect_invalid(
      {"program", "--model", model.c_str(), "--shape", "1x1", "--types",
       "float", "--pool-size", "-1"});
}
