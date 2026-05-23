#include <ATen/core/ScalarType.h>
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "grpc/client/client_args.hpp"
#include "grpc/client/inference_client.hpp"
#include "grpc/server/inference_service.hpp"
#include "support/grpc/client/inference_client_test_api.hpp"
#include "test_helpers.hpp"

TEST(ClientArgs, ParsesValidArguments)
{
  auto argv = std::to_array<const char*>(
      {"prog", "--input", "input:1x3x224x224:float32", "--server",
       "localhost:1234", "--model", "my_model", "--request-number", "5",
       "--delay", "10", "--schedule-csv", "/tmp/schedule.csv", "--summary-json",
       "/tmp/client-summary.json", "--verbose", "2"});
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  EXPECT_TRUE(cfg.valid);
  ASSERT_EQ(cfg.inputs.size(), 1U);
  EXPECT_EQ(cfg.inputs[0].shape, (std::vector<int64_t>{1, 3, 224, 224}));
  EXPECT_EQ(cfg.inputs[0].type, at::kFloat);
  EXPECT_EQ(cfg.server_address, "localhost:1234");
  EXPECT_EQ(cfg.model_name, "my_model");
  EXPECT_EQ(cfg.model_version, "1");
  EXPECT_EQ(cfg.request_nb, 5);
  EXPECT_TRUE(cfg.request_nb_explicit);
  EXPECT_EQ(cfg.delay_us, 10);
  EXPECT_EQ(cfg.schedule_csv_path, "/tmp/schedule.csv");
  EXPECT_EQ(cfg.summary_json_path, "/tmp/client-summary.json");
  EXPECT_EQ(cfg.verbosity, starpu_server::VerbosityLevel::Stats);
}

TEST(ClientArgs, HelpFlagSetsShowHelp)
{
  const auto help_flags = std::to_array<const char*>({"--help", "-h"});
  for (const auto* flag : help_flags) {
    SCOPED_TRACE(flag);
    auto argv = std::to_array<const char*>({"prog", flag});
    auto cfg = starpu_server::parse_client_args(std::span{argv});
    EXPECT_TRUE(cfg.show_help);
    EXPECT_TRUE(cfg.valid);
    EXPECT_TRUE(cfg.inputs.empty());
  }
}

TEST(ClientArgs, MissingInputMarksConfigInvalid)
{
  auto argv = std::to_array<const char*>({"prog"});
  testing::internal::CaptureStderr();
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(cfg.valid);
  EXPECT_TRUE(cfg.inputs.empty());
  EXPECT_NE(err.find("--input option is required."), std::string::npos);
}

TEST(ClientArgs, RejectsUnknownArguments)
{
  auto argv = std::to_array<const char*>({"prog", "--bogus"});
  testing::internal::CaptureStderr();
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(cfg.valid);
  EXPECT_NE(err.find("Unknown argument"), std::string::npos);
}

TEST(ClientArgs, RejectsMalformedInputFormat)
{
  auto argv = std::to_array<const char*>({"prog", "--input", "bad"});
  testing::internal::CaptureStderr();
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(cfg.valid);
  EXPECT_NE(err.find("Input must be NAME:SHAPE:TYPE"), std::string::npos);
}

TEST(ClientArgs, ParsesClientModelPathWhenProvided)
{
  auto argv = std::to_array<const char*>(
      {"prog", "--input", "input:1:float32", "--client-model",
       "/tmp/model.pt"});
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  ASSERT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.client_model_path, "/tmp/model.pt");
}

TEST(ClientArgs, ParsesModelVersionWhenProvided)
{
  auto argv = std::to_array<const char*>(
      {"prog", "--input", "input:1:float32", "--model-version", "42"});
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  ASSERT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.model_version, "42");
}

TEST(ClientArgs, MissingClientModelPathValueFailsParsing)
{
  auto argv = std::to_array<const char*>(
      {"prog", "--input", "input:1:float32", "--client-model"});
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  EXPECT_FALSE(cfg.valid);
  EXPECT_TRUE(cfg.client_model_path.empty());
}

TEST(ClientArgs, MissingModelVersionValueFailsParsing)
{
  auto argv = std::to_array<const char*>(
      {"prog", "--input", "input:1:float32", "--model-version"});
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  EXPECT_FALSE(cfg.valid);
  EXPECT_EQ(cfg.model_version, "1");
}

TEST(ClientArgs, ParsesScheduleCsvPathWhenProvided)
{
  auto argv = std::to_array<const char*>(
      {"prog", "--input", "input:1:float32", "--schedule-csv",
       "/tmp/traffic.csv"});
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  ASSERT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.schedule_csv_path, "/tmp/traffic.csv");
}

TEST(ClientArgs, MissingScheduleCsvValueFailsParsing)
{
  auto argv = std::to_array<const char*>(
      {"prog", "--input", "input:1:float32", "--schedule-csv"});
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  EXPECT_FALSE(cfg.valid);
  EXPECT_TRUE(cfg.schedule_csv_path.empty());
}

TEST(ClientArgs, ParsesSummaryJsonPathWhenProvided)
{
  auto argv = std::to_array<const char*>(
      {"prog", "--input", "input:1:float32", "--summary-json",
       "/tmp/perf-summary.json"});
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  ASSERT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.summary_json_path, "/tmp/perf-summary.json");
}

TEST(ClientArgs, MissingSummaryJsonValueFailsParsing)
{
  auto argv = std::to_array<const char*>(
      {"prog", "--input", "input:1:float32", "--summary-json"});
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  EXPECT_FALSE(cfg.valid);
  EXPECT_TRUE(cfg.summary_json_path.empty());
}

TEST(ClientArgs, RejectsNegativeDelay)
{
  auto argv = std::to_array<const char*>(
      {"prog", "--input", "input:1x3:float32", "--delay", "-1"});
  testing::internal::CaptureStderr();
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(cfg.valid);
  EXPECT_NE(err.find("Must be >= 0."), std::string::npos);
}

TEST(ClientArgs, VerboseLevels)
{
  using enum starpu_server::VerbosityLevel;
  const auto cases =
      std::to_array<std::pair<const char*, starpu_server::VerbosityLevel>>(
          {{"0", Silent}, {"1", Info}, {"3", Debug}, {"4", Trace}});
  for (const auto& [level_str, expected] : cases) {
    auto argv = std::to_array<const char*>(
        {"prog", "--input", "input:1:float32", "--verbose", level_str});
    auto cfg = starpu_server::parse_client_args(std::span{argv});
    ASSERT_TRUE(cfg.valid);
    EXPECT_EQ(cfg.verbosity, expected);
  }
}

TEST(ClientArgs, RejectsNonPositiveShapeDims)
{
  auto argv_neg =
      std::to_array<const char*>({"prog", "--input", "input:1x-3x224:float32"});
  testing::internal::CaptureStderr();
  auto cfg_neg = starpu_server::parse_client_args(std::span{argv_neg});
  const std::string neg_err = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(cfg_neg.valid);
  EXPECT_NE(
      neg_err.find("Shape contains non-positive dimension: -3"),
      std::string::npos);

  auto argv_zero =
      std::to_array<const char*>({"prog", "--input", "input:1x0x224:float32"});
  testing::internal::CaptureStderr();
  auto cfg_zero = starpu_server::parse_client_args(std::span{argv_zero});
  const std::string zero_err = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(cfg_zero.valid);
  EXPECT_NE(
      zero_err.find("Shape contains non-positive dimension: 0"),
      std::string::npos);
}

TEST(ClientArgs, RejectsMalformedShapeTokens)
{
  const auto cases =
      std::to_array<const char*>({"1xax2", "9223372036854775808", ""});
  for (const auto* shape : cases) {
    const std::string spec = std::string{"input:"} + shape + ":float32";
    auto argv = std::to_array<const char*>({"prog", "--input", spec.c_str()});
    testing::internal::CaptureStderr();
    auto cfg = starpu_server::parse_client_args(std::span{argv});
    const std::string err = testing::internal::GetCapturedStderr();
    EXPECT_FALSE(cfg.valid);
    EXPECT_FALSE(err.empty());
  }
}

TEST(ClientArgsHelp, ContainsKeyOptions)
{
  testing::internal::CaptureStdout();
  starpu_server::display_client_help("prog");
  const std::string out = testing::internal::GetCapturedStdout();
  EXPECT_NE(out.find("--request-number"), std::string::npos);
  EXPECT_NE(out.find("--delay"), std::string::npos);
  EXPECT_NE(out.find("--schedule-csv"), std::string::npos);
  EXPECT_NE(out.find("--summary-json"), std::string::npos);
  EXPECT_NE(out.find("--input"), std::string::npos);
  EXPECT_NE(out.find("--server"), std::string::npos);
  EXPECT_NE(out.find("--model"), std::string::npos);
  EXPECT_NE(out.find("--model-version"), std::string::npos);
  EXPECT_NE(out.find("--client-model"), std::string::npos);
  EXPECT_NE(out.find("--verbose"), std::string::npos);
  EXPECT_NE(out.find("--help"), std::string::npos);
  EXPECT_EQ(out.find("--shape"), std::string::npos);
  EXPECT_EQ(out.find("--type"), std::string::npos);
}

TEST(InferenceClientDetermineInferenceCount, HandlesEdgeCases)
{
  const auto determine = [](const starpu_server::ClientConfig& cfg) {
    return starpu_server::InferenceClientTestAccess::determine_inference_count(
        cfg);
  };

  starpu_server::ClientConfig cfg;
  EXPECT_EQ(determine(cfg), 1U);

  cfg.inputs.clear();
  {
    starpu_server::InputConfig input;
    input.name = "zero_batch";
    input.shape = {0, 3, 224};
    cfg.inputs.push_back(input);
  }
  {
    starpu_server::InputConfig input;
    input.name = "negative_batch";
    input.shape = {-4, 3, 224};
    cfg.inputs.push_back(input);
  }
  {
    starpu_server::InputConfig input;
    input.name = "valid_batch";
    input.shape = {5, 3, 224};
    cfg.inputs.push_back(input);
  }
  EXPECT_EQ(determine(cfg), 5U);

  cfg.inputs.clear();
  {
    starpu_server::InputConfig input;
    input.name = "first_valid";
    input.shape = {8, 3};
    cfg.inputs.push_back(input);
  }
  {
    starpu_server::InputConfig input;
    input.name = "conflicting";
    input.shape = {3, 3};
    cfg.inputs.push_back(input);
  }
  testing::internal::CaptureStderr();
  EXPECT_EQ(determine(cfg), 8U);
  const std::string warning = testing::internal::GetCapturedStderr();
  EXPECT_NE(
      warning.find("Inconsistent batch dimension across inputs (8 vs 3)"),
      std::string::npos);

  cfg.inputs.clear();
  {
    starpu_server::InputConfig input;
    input.name = "empty_shape";
    cfg.inputs.push_back(input);
  }
  {
    starpu_server::InputConfig input;
    input.name = "zero_dim";
    input.shape = {0};
    cfg.inputs.push_back(input);
  }
  {
    starpu_server::InputConfig input;
    input.name = "negative_dim";
    input.shape = {-1, 2};
    cfg.inputs.push_back(input);
  }
  EXPECT_EQ(determine(cfg), 1U);
}

class ParseInputTypeCase
    : public ::testing::TestWithParam<std::pair<const char*, at::ScalarType>> {
};

TEST_P(ParseInputTypeCase, ParsesExpectedType)
{
  const auto& [type_str, expected] = GetParam();
  const std::string arg = std::string{"input:1:"} + type_str;
  auto argv = std::to_array<const char*>({"prog", "--input", arg.c_str()});
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  ASSERT_TRUE(cfg.valid);
  ASSERT_EQ(cfg.inputs.size(), 1U);
  EXPECT_EQ(cfg.inputs[0].type, expected);
}

INSTANTIATE_TEST_SUITE_P(
    SupportedTypes, ParseInputTypeCase,
    ::testing::Values(
        std::pair{"float32", at::kFloat}, std::pair{"float64", at::kDouble},
        std::pair{"float16", at::kHalf}, std::pair{"bfloat16", at::kBFloat16},
        std::pair{"int32", at::kInt}, std::pair{"int64", at::kLong},
        std::pair{"int16", at::kShort}, std::pair{"int8", at::kChar},
        std::pair{"uint8", at::kByte}, std::pair{"bool", at::kBool}));


TEST(InferenceClient, ModelIsReadyReturnsTrue)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  auto server = starpu_server::start_test_grpc_server(queue, reference_outputs);

  auto channel = grpc::CreateChannel(
      "127.0.0.1:" + std::to_string(server.port),
      grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);
  EXPECT_TRUE(client.ModelIsReady({"example", "1"}));

  starpu_server::StopServer(server.server.get());
  server.thread.join();
  EXPECT_EQ(server.server, nullptr);
}

TEST(InferenceClient, ModelIsReadyReturnsFalseWhenUnavailable)
{
  auto channel = grpc::CreateChannel(
      "127.0.0.1:59997", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);
  testing::internal::CaptureStderr();
  EXPECT_FALSE(client.ModelIsReady({"example", "1"}));
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_NE(err.find("RPC failed"), std::string::npos);
}

TEST(InferenceClientLatencySummary, SkipsEmptyMetric)
{
  auto channel = grpc::CreateChannel(
      "localhost:59998", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Info);

  starpu_server::InferenceClientTestAccess::set_verbosity(
      client, starpu_server::VerbosityLevel::Info);
  auto& records =
      starpu_server::InferenceClientTestAccess::latency_records(client);
  records.roundtrip_ms.push_back(1.23);
  records.server_queue_ms.push_back(0.45);

  testing::internal::CaptureStdout();
  starpu_server::InferenceClientTestAccess::log_latency_summary(client);
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("latency"), std::string::npos);
  EXPECT_NE(output.find("queue"), std::string::npos);
  EXPECT_EQ(output.find("client_overhead"), std::string::npos);
}

TEST(InferenceClientLatencySummary, HandlesZeroElapsedTime)
{
  auto channel = grpc::CreateChannel(
      "localhost:59995", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Stats);

  const auto now = std::chrono::system_clock::now();
  starpu_server::InferenceClientTestAccess::set_first_request_time(client, now);
  starpu_server::InferenceClientTestAccess::set_last_response_time(client, now);
  starpu_server::InferenceClientTestAccess::set_total_inference_count(
      client, 3);
  const starpu_server::InferenceClientTestAccess::LatencySample sample{
      1.0, 0.9, 0.8, 0.7,  0.65, 0.6,  0.5, 0.4,
      0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02};
  starpu_server::InferenceClientTestAccess::record_latency(client, sample);

  starpu_server::CaptureStream capture{std::cout};
  starpu_server::InferenceClientTestAccess::log_latency_summary(client);
  const std::string output = capture.str();

  EXPECT_NE(
      output.find(
          "throughput: 3 inferences (elapsed time too small to compute rate)"),
      std::string::npos);
}

TEST(InferenceClientSummaryJson, WritesExpectedFields)
{
  auto channel = grpc::CreateChannel(
      "localhost:59994", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);

  const auto start =
      std::chrono::system_clock::time_point{std::chrono::seconds{10}};
  const auto end = start + std::chrono::milliseconds(250);
  starpu_server::InferenceClientTestAccess::set_first_request_time(
      client, start);
  starpu_server::InferenceClientTestAccess::set_last_response_time(client, end);
  starpu_server::InferenceClientTestAccess::set_total_inference_count(
      client, 5);
  starpu_server::InferenceClientTestAccess::set_total_requests_sent(client, 4);
  starpu_server::InferenceClientTestAccess::set_success_requests(client, 3);
  starpu_server::InferenceClientTestAccess::set_rejected_requests(client, 1);

  const starpu_server::InferenceClientTestAccess::LatencySample sample{
      3.0, 2.5, 0.8,  0.7, 0.6,  0.5,  0.4, 0.3,
      0.2, 0.1, 0.05, 1.5, 0.04, 0.03, 0.02};
  starpu_server::InferenceClientTestAccess::record_latency(client, sample);

  const auto summary_path =
      std::filesystem::temp_directory_path() /
      std::format(
          "starpu_client_summary_test_{}.json",
          std::chrono::steady_clock::now().time_since_epoch().count());
  std::error_code remove_ec;
  std::filesystem::remove(summary_path, remove_ec);

  ASSERT_TRUE(client.write_summary_json(summary_path));

  std::ifstream stream(summary_path);
  ASSERT_TRUE(stream.is_open());
  const std::string json{
      std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};

  EXPECT_NE(
      json.find("\"requests\": {\"sent\": 4, \"handled\": 4"),
      std::string::npos);
  EXPECT_NE(json.find("\"ok\": 3"), std::string::npos);
  EXPECT_NE(json.find("\"rejected\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"inference_count\": 5"), std::string::npos);
  EXPECT_NE(json.find("\"response_count\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"elapsed_seconds\": 0.25"), std::string::npos);
  EXPECT_NE(json.find("\"throughput_rps\": 20"), std::string::npos);
  EXPECT_NE(json.find("\"roundtrip\": {\"mean_ms\": 3"), std::string::npos);
  EXPECT_NE(
      json.find("\"server_queue\": {\"mean_ms\": 0.7"), std::string::npos);

  std::filesystem::remove(summary_path, remove_ec);
}

TEST(InferenceClientSummaryJson, WritesNullLatencyFieldsWhenNoResponsesRecorded)
{
  auto channel = grpc::CreateChannel(
      "localhost:59993", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);

  const auto summary_path =
      std::filesystem::temp_directory_path() /
      std::format(
          "starpu_client_summary_nulls_{}.json",
          std::chrono::steady_clock::now().time_since_epoch().count());
  std::error_code remove_ec;
  std::filesystem::remove(summary_path, remove_ec);

  ASSERT_TRUE(client.write_summary_json(summary_path));

  std::ifstream stream(summary_path);
  ASSERT_TRUE(stream.is_open());
  const std::string json{
      std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};

  EXPECT_NE(json.find("\"roundtrip\": null"), std::string::npos);
  EXPECT_NE(json.find("\"server_overall\": null"), std::string::npos);
  EXPECT_NE(json.find("\"client_overhead\": null"), std::string::npos);

  std::filesystem::remove(summary_path, remove_ec);
}

TEST(InferenceClientSummaryJson, ReturnsFalseWhenCreatingParentDirectoryFails)
{
  auto channel = grpc::CreateChannel(
      "localhost:59992", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);

  const auto blocker_path =
      std::filesystem::temp_directory_path() /
      std::format(
          "starpu_client_summary_blocker_{}",
          std::chrono::steady_clock::now().time_since_epoch().count());
  {
    std::ofstream blocker(blocker_path);
    ASSERT_TRUE(blocker.is_open());
    blocker << "not a directory";
  }

  const auto summary_path = blocker_path / "nested" / "summary.json";
  testing::internal::CaptureStderr();
  EXPECT_FALSE(client.write_summary_json(summary_path));
  const std::string err = testing::internal::GetCapturedStderr();

  EXPECT_NE(err.find("Failed to create summary directory"), std::string::npos);

  std::error_code remove_ec;
  std::filesystem::remove(blocker_path, remove_ec);
}

TEST(InferenceClientSummaryJson, ReturnsFalseWhenSummaryPathCannotBeOpened)
{
  auto channel = grpc::CreateChannel(
      "localhost:59991", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);

  const auto summary_dir =
      std::filesystem::temp_directory_path() /
      std::format(
          "starpu_client_summary_dir_{}",
          std::chrono::steady_clock::now().time_since_epoch().count());
  std::filesystem::create_directories(summary_dir);

  testing::internal::CaptureStderr();
  EXPECT_FALSE(client.write_summary_json(summary_dir));
  const std::string err = testing::internal::GetCapturedStderr();

  EXPECT_NE(err.find("Failed to open summary JSON file"), std::string::npos);

  std::error_code remove_ec;
  std::filesystem::remove_all(summary_dir, remove_ec);
}

TEST(InferenceClient, RejectsMismatchedTensorCount)
{
  auto channel =
      grpc::CreateChannel("localhost:0", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);

  starpu_server::ClientConfig cfg;
  cfg.model_name = "example";
  cfg.model_version = "1";
  cfg.inputs.push_back({"input0", {1}, at::kFloat});

  std::vector<torch::Tensor> tensors = {torch::zeros({1}), torch::zeros({1})};

  testing::internal::CaptureStderr();
  EXPECT_THROW(client.AsyncModelInfer(tensors, cfg), std::invalid_argument);
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_NE(
      err.find("Mismatched number of input tensors: expected 1, got 2"),
      std::string::npos);
}

TEST(InferenceClient, RejectsUnsupportedTensorType)
{
  auto channel =
      grpc::CreateChannel("localhost:0", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);

  starpu_server::ClientConfig cfg;
  cfg.model_name = "example";
  cfg.model_version = "1";
  cfg.inputs.push_back({"input0", {1}, at::kFloat});

  std::vector<torch::Tensor> tensors = {
      torch::zeros({1}, torch::dtype(at::kDouble))};

  testing::internal::CaptureStderr();
  EXPECT_THROW(
      {
        try {
          client.AsyncModelInfer(tensors, cfg);
          client.Shutdown();
        }
        catch (...) {
          client.Shutdown();
          throw;
        }
      },
      std::invalid_argument);
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_NE(
      err.find(
          "Unsupported tensor type for input input0: expected FP32, got FP64"),
      std::string::npos);
}

TEST(InferenceClient, ConvertsNonContiguousCpuTensor)
{
  auto channel =
      grpc::CreateChannel("localhost:0", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Info);

  starpu_server::ClientConfig cfg;
  cfg.model_name = "example";
  cfg.model_version = "1";
  cfg.inputs.push_back({"input0", {2, 2}, at::kFloat});

  auto tensor = torch::arange(4, torch::kFloat).view({2, 2}).transpose(0, 1);

  testing::internal::CaptureStdout();
  try {
    client.AsyncModelInfer({tensor}, cfg);
    client.Shutdown();
  }
  catch (...) {
    client.Shutdown();
    testing::internal::GetCapturedStdout();
    throw;
  }
  const std::string output = testing::internal::GetCapturedStdout();
  EXPECT_NE(
      output.find("not on CPU or non-contiguous, converting"),
      std::string::npos);
}

class ParseVerbosityLevelValid
    : public ::testing::TestWithParam<
          std::pair<const char*, starpu_server::VerbosityLevel>> {};

TEST_P(ParseVerbosityLevelValid, ReturnsExpectedEnum)
{
  const auto& [input, expected] = GetParam();
  EXPECT_EQ(starpu_server::parse_verbosity_level(input), expected);
}

INSTANTIATE_TEST_SUITE_P(
    ParseVerbosityLevel, ParseVerbosityLevelValid,
    ::testing::Values(
        std::pair{"0", starpu_server::VerbosityLevel::Silent},
        std::pair{"1", starpu_server::VerbosityLevel::Info},
        std::pair{"2", starpu_server::VerbosityLevel::Stats},
        std::pair{"3", starpu_server::VerbosityLevel::Debug},
        std::pair{"4", starpu_server::VerbosityLevel::Trace}));

TEST(InferenceClient, AsyncCompleteRpcExitsAfterShutdown)
{
  auto channel = grpc::CreateChannel(
      "127.0.0.1:59996", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);

  std::thread cq_thread(
      &starpu_server::InferenceClient::AsyncCompleteRpc, &client);
  constexpr int kSleepMs = 50;
  std::this_thread::sleep_for(std::chrono::milliseconds(kSleepMs));
  client.Shutdown();
  cq_thread.join();
  SUCCEED();
}
