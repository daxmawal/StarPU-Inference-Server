#include <gtest/gtest.h>
#include <torch/torch.h>

#include <format>
#include <limits>
#include <string>
#include <vector>

#include "cli/args_parser.hpp"
#include "test_cli.hpp"
#include "test_helpers.hpp"
#include "utils/exceptions.hpp"

static auto
build_common_args() -> std::vector<const char*>
{
  return {"program", "--model", test_model_path().c_str(), "--shape", "1x3",
          "--types", "float"};
}

class ArgsParserInvalidOptions_Robustesse
    : public ::testing::TestWithParam<std::vector<const char*>> {};

TEST_P(ArgsParserInvalidOptions_Robustesse, Invalid)
{
  auto args = build_common_args();
  const auto& diff = GetParam();
  args.insert(args.end(), diff.begin(), diff.end());
  expect_invalid(args);
}

struct TypesParseErrorParam {
  const char* types_value;
  const char* expected_message;
};

class ArgsParserTypesParseErrors_Robustesse
    : public ::testing::TestWithParam<TypesParseErrorParam> {};

TEST_P(ArgsParserTypesParseErrors_Robustesse, ReportsError)
{
  const auto& param = GetParam();
  std::vector<const char*> args = {
      "program", "--model", test_model_path().c_str(), "--shape",
      "1x3",     "--types", param.types_value};

  starpu_server::CaptureStream capture{std::cerr};
  const auto result = parse(args);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(
      capture.str(), starpu_server::expected_log_line(
                         starpu_server::ErrorLevel, param.expected_message));
}

INSTANTIATE_TEST_SUITE_P(
    TypesParseErrors, ArgsParserTypesParseErrors_Robustesse,
    ::testing::Values(
        TypesParseErrorParam{"float,", "Trailing comma in types string"},
        TypesParseErrorParam{"float,,int", "Empty type in types string"},
        TypesParseErrorParam{",float", "Empty type in types string"},
        TypesParseErrorParam{",", "No types provided."}));

INSTANTIATE_TEST_SUITE_P(
    InvalidArguments, ArgsParserInvalidOptions_Robustesse,
    ::testing::Values(
        std::vector<const char*>{"--types", ""},
        std::vector<const char*>{"--request-number", "0"},
        std::vector<const char*>{"--shape", "1x-3x3"},
        std::vector<const char*>{"--types", "unknown"},
        std::vector<const char*>{"--types", "complex64"},
        std::vector<const char*>{"--scheduler", "unknown"},
        std::vector<const char*>{"--device-ids", "-1"},
        std::vector<const char*>{"--device-ids", "0,0"},
        std::vector<const char*>{"--unknown"},
        std::vector<const char*>{"--shapes", "1x2,2x3", "--types", "float"},
        std::vector<const char*>{"--shape", "1xax2"},
        std::vector<const char*>{"--shape", "9223372036854775808"},
        std::vector<const char*>{"--verbose", "5"},
        std::vector<const char*>{"--shape", ""},
        std::vector<const char*>{"--shapes", ""},
        std::vector<const char*>{"--shapes", "1x2,", "--types", "float"},
        std::vector<const char*>{"--request-number", "-1"},
        std::vector<const char*>{"--delay", "-1"},
        std::vector<const char*>{"--max-batch-size", "0"},
        std::vector<const char*>{"--max-batch-size", "-1"},
        std::vector<const char*>{"--metrics-port", "0"},
        std::vector<const char*>{"--metrics-port", "65536"},
        std::vector<const char*>{"--pregen-inputs", "0"},
        std::vector<const char*>{"--pregen-inputs", "-1"},
        std::vector<const char*>{"--warmup-request_nb", "-1"},
        std::vector<const char*>{"--seed", "-1"},
        std::vector<const char*>{"--rtol", "-1"},
        std::vector<const char*>{"--atol", "-1"},
        std::vector<const char*>{"--shapes", "1x2,,3", "--types", "float,int"},
        std::vector<const char*>{"--model"},
        std::vector<const char*>{"--request-number"},
        std::vector<const char*>{"--config"},
        std::vector<const char*>{"--scheduler"},
        std::vector<const char*>{"--address"},
        std::vector<const char*>{"--model", "/nonexistent/path/to/model.pt"},
        std::vector<const char*>{
            "--config", "/nonexistent/path/to/config.yaml"}));

TEST(ArgsParserInvalidOptions_Robustesse, DeviceIdOutOfRange)
{
  const int device_count =
      static_cast<int>(static_cast<unsigned char>(torch::cuda::device_count()));
  std::string id_str = std::to_string(device_count);
  auto args = build_common_args();
  args.emplace_back("--device-ids");
  args.push_back(id_str.c_str());
  starpu_server::CaptureStream capture{std::cerr};
  const auto opts = parse(args);
  EXPECT_FALSE(opts.valid);
  const std::string expected_msg = std::format(
      "GPU ID {} out of range. Only {} device(s) available.", device_count,
      device_count);
  EXPECT_EQ(
      capture.str(), starpu_server::expected_log_line(
                         starpu_server::ErrorLevel, expected_msg));
}

TEST(ArgsParserInvalidOptions_Robustesse, TypesCountMustMatchShapes)
{
  const std::vector<const char*> args = {
      "program", "--model",   test_model_path().c_str(), "--shape", "1x3",
      "--types", "float,int",
  };

  starpu_server::CaptureStream capture{std::cerr};
  const auto opts = parse(args);

  EXPECT_FALSE(opts.valid);
  EXPECT_EQ(
      capture.str(),
      starpu_server::expected_log_line(
          starpu_server::ErrorLevel,
          "Number of --types must match number of input shapes."));
}

struct MissingValueParam {
  std::vector<const char*> args;
  const char* option;
};

class ArgsParserMissingValues_Robustesse
    : public ::testing::TestWithParam<MissingValueParam> {};

TEST_P(ArgsParserMissingValues_Robustesse, ReportsError)
{
  const auto& param = GetParam();
  starpu_server::CaptureStream capture{std::cerr};
  const auto opts = parse(param.args);
  EXPECT_FALSE(opts.valid);
  const std::string expected_msg =
      std::string(param.option) + " option requires a value.";
  EXPECT_EQ(
      capture.str(), starpu_server::expected_log_line(
                         starpu_server::ErrorLevel, expected_msg));
}

INSTANTIATE_TEST_SUITE_P(
    MissingValue, ArgsParserMissingValues_Robustesse,
    ::testing::Values(
        MissingValueParam{{"program", "--model"}, "--model"},
        MissingValueParam{{"program", "--config"}, "--config"},
        MissingValueParam{{"program", "--scheduler"}, "--scheduler"},
        MissingValueParam{{"program", "--address"}, "--address"},
        MissingValueParam{{"program", "--rtol"}, "--rtol"},
        MissingValueParam{{"program", "--atol"}, "--atol"}));

TEST(
    ArgsParserComputeMaxMessageBytes_Robustesse,
    ReportsInvalidDimensionException)
{
  auto argv = build_argv({"program"});
  starpu_server::RuntimeConfig opts;
  opts.models.resize(1);
  auto& model = opts.models.front();
  model.path = test_model_path();
  model.inputs.emplace_back();
  auto& input = model.inputs.back();
  input.name = "input";
  input.dims = {1, -1};
  input.type = at::kFloat;
  model.outputs.emplace_back();
  auto& output = model.outputs.back();
  output.name = "output";
  output.dims = {1};
  output.type = at::kFloat;

  starpu_server::CaptureStream capture{std::cerr};
  const auto result =
      starpu_server::parse_arguments({argv.data(), argv.size()}, opts);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(
      capture.str(),
      starpu_server::expected_log_line(
          starpu_server::ErrorLevel, "dimension size must be non-negative"));
}

TEST(
    ArgsParserComputeMaxMessageBytes_Robustesse,
    ReportsMessageSizeOverflowException)
{
  auto argv = build_argv({"program"});
  starpu_server::RuntimeConfig opts;
  opts.models.resize(1);
  auto& model = opts.models.front();
  model.path = test_model_path();
  model.inputs.emplace_back();
  auto& input = model.inputs.back();
  input.name = "input";
  input.dims = {std::numeric_limits<int64_t>::max()};
  input.type = at::kFloat;
  model.outputs.emplace_back();
  auto& output = model.outputs.back();
  output.name = "output";
  output.dims = {1};
  output.type = at::kFloat;

  starpu_server::CaptureStream capture{std::cerr};
  const auto result =
      starpu_server::parse_arguments({argv.data(), argv.size()}, opts);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(
      capture.str(), starpu_server::expected_log_line(
                         starpu_server::ErrorLevel,
                         "numel * element size would overflow size_t"));
}
