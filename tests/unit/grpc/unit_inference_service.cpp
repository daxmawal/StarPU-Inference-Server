#include <prometheus/metric_family.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>

#include "monitoring/metrics.hpp"
#include "test_constants.hpp"
#include "test_inference_service.hpp"
#include "utils/batching_trace_logger.hpp"

namespace {
using starpu_server::test_constants::kF1;
using starpu_server::test_constants::kF2;
using starpu_server::test_constants::kF3;
using starpu_server::test_constants::kF4;
using starpu_server::test_constants::kI10;
using starpu_server::test_constants::kI20;
using starpu_server::test_constants::kI30;

auto
make_temp_trace_path() -> std::filesystem::path
{
  static std::atomic<std::uint64_t> trace_file_counter{0};
  const auto suffix =
      trace_file_counter.fetch_add(1, std::memory_order_relaxed);
  return std::filesystem::temp_directory_path() /
         ("submit_job_trace_" + std::to_string(suffix) + ".json");
}

struct TraceFileGuard {
  starpu_server::BatchingTraceLogger& tracer;
  std::filesystem::path path;

  ~TraceFileGuard()
  {
    tracer.configure(false, "");
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

struct HandleModelInferAsyncHooksGuard {
  explicit HandleModelInferAsyncHooksGuard(
      starpu_server::InferenceServiceImpl::HandleModelInferAsyncTestHooks hooks)
  {
    starpu_server::InferenceServiceImpl::TestAccessor::
        SetHandleModelInferAsyncTestHooks(std::move(hooks));
  }

  ~HandleModelInferAsyncHooksGuard()
  {
    starpu_server::InferenceServiceImpl::TestAccessor::
        ClearHandleModelInferAsyncTestHooks();
  }
};

struct HandleAsyncInferCompletionHooksGuard {
  explicit HandleAsyncInferCompletionHooksGuard(
      starpu_server::InferenceServiceImpl::HandleAsyncInferCompletionTestHooks
          hooks)
  {
    starpu_server::InferenceServiceImpl::TestAccessor::
        SetHandleAsyncInferCompletionTestHooks(std::move(hooks));
  }

  ~HandleAsyncInferCompletionHooksGuard()
  {
    starpu_server::InferenceServiceImpl::TestAccessor::
        ClearHandleAsyncInferCompletionTestHooks();
  }
};

struct ModelStatisticsNullTargetGuard {
  explicit ModelStatisticsNullTargetGuard(bool enable = true)
  {
    starpu_server::InferenceServiceImpl::TestAccessor::
        SetModelStatisticsForceNullTargetForTest(enable);
  }

  ~ModelStatisticsNullTargetGuard()
  {
    starpu_server::InferenceServiceImpl::TestAccessor::
        SetModelStatisticsForceNullTargetForTest(false);
  }
};

auto
sum_counter_values_for_label(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view family_name, std::string_view label_name,
    std::string_view label_value) -> double
{
  double sum = 0.0;
  for (const auto& family : families) {
    if (family.name != family_name) {
      continue;
    }
    for (const auto& metric : family.metric) {
      const bool has_label =
          std::ranges::any_of(metric.label, [&](const auto& label) {
            return label.name == label_name && label.value == label_value;
          });
      if (has_label) {
        sum += metric.counter.value;
      }
    }
  }
  return sum;
}
}  // namespace

class MetricsInferenceServiceTest : public InferenceServiceTest {
 protected:
  void SetUp() override
  {
    starpu_server::shutdown_metrics();
    ASSERT_TRUE(starpu_server::init_metrics(0));
    InferenceServiceTest::SetUp();
  }

  void TearDown() override
  {
    starpu_server::shutdown_metrics();
    InferenceServiceTest::TearDown();
  }
};

class MultiTypeInferenceServiceTest : public InferenceServiceTest {
 protected:
  [[nodiscard]] auto make_service_config() const -> ServiceConfig override
  {
    ServiceConfig config;
    config.expected_input_types = {at::kFloat, at::kLong};
    return config;
  }
};

class NamedInputInferenceServiceTest : public InferenceServiceTest {
 protected:
  void SetUp() override
  {
    std::vector<at::ScalarType> expected_types = {at::kFloat, at::kLong};
    std::vector<std::string> expected_names = {"first", "second"};
    service = std::make_unique<starpu_server::InferenceServiceImpl>(
        &queue, &ref_outputs, std::move(expected_types),
        starpu_server::InferenceServiceImpl::ServiceOptions{
            .expected_input_names = std::move(expected_names)});
  }
};

class LongInputInferenceServiceTest : public InferenceServiceTest {
 protected:
  [[nodiscard]] auto make_service_config() const -> ServiceConfig override
  {
    ServiceConfig config;
    config.expected_input_types = {at::kLong};
    return config;
  }
};

class ConfiguredShapeNoBatchingInferenceServiceTest
    : public InferenceServiceTest {
 protected:
  [[nodiscard]] auto make_service_config() const -> ServiceConfig override
  {
    ServiceConfig config;
    config.expected_input_dims = std::vector<std::vector<int64_t>>{{2, 2}};
    config.max_batch_size = 0;
    return config;
  }
};

class ConfiguredShapeWithBatchingInferenceServiceTest
    : public InferenceServiceTest {
 protected:
  [[nodiscard]] auto make_service_config() const -> ServiceConfig override
  {
    ServiceConfig config;
    config.expected_input_dims = std::vector<std::vector<int64_t>>{{2, 2}};
    config.max_batch_size = 4;
    return config;
  }
};

class ZeroRankConfiguredShapeWithBatchingInferenceServiceTest
    : public InferenceServiceTest {
 protected:
  [[nodiscard]] auto make_service_config() const -> ServiceConfig override
  {
    ServiceConfig config;
    config.expected_input_dims = std::vector<std::vector<int64_t>>{{}};
    config.max_batch_size = 4;
    return config;
  }
};

TEST(ComputeThreadCount, ZeroConcurrencyDefaults)
{
  EXPECT_EQ(
      starpu_server::compute_thread_count_from(0U),
      starpu_server::kDefaultGrpcThreads);
}

TEST(ComputeThreadCount, ClampsToConfiguredBounds)
{
  EXPECT_EQ(
      starpu_server::compute_thread_count_from(1U),
      starpu_server::kMinGrpcThreads);
  EXPECT_EQ(
      starpu_server::compute_thread_count_from(100U),
      starpu_server::kMaxGrpcThreads);
}

TEST(InferenceServiceImpl, ServerMetadataUsesServerNameAndVersion)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, {at::kFloat},
      starpu_server::InferenceServiceImpl::ServiceOptions{
          .default_model_name = "default_model",
          .server_name = "my_server",
          .server_version = "1.2.3"});

  grpc::ServerContext ctx;
  inference::ServerMetadataRequest req;
  inference::ServerMetadataResponse reply;

  auto status = service.ServerMetadata(&ctx, &req, &reply);

  ASSERT_TRUE(status.ok());
  EXPECT_EQ(reply.name(), "my_server");
  EXPECT_EQ(reply.version(), "1.2.3");
}

TEST(InferenceServiceImpl, ServerMetadataFallsBackToDefaultModelName)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, {at::kFloat},
      starpu_server::InferenceServiceImpl::ServiceOptions{
          .default_model_name = "fallback_model"});

  grpc::ServerContext ctx;
  inference::ServerMetadataRequest req;
  inference::ServerMetadataResponse reply;

  auto status = service.ServerMetadata(&ctx, &req, &reply);

  ASSERT_TRUE(status.ok());
  EXPECT_EQ(reply.name(), "fallback_model");
  EXPECT_TRUE(reply.version().empty());
}

TEST(InferenceServiceImpl, ServerMetadataUsesHardcodedFallbackName)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, {at::kFloat});

  grpc::ServerContext ctx;
  inference::ServerMetadataRequest req;
  inference::ServerMetadataResponse reply;

  auto status = service.ServerMetadata(&ctx, &req, &reply);

  ASSERT_TRUE(status.ok());
  EXPECT_EQ(reply.name(), "starpu_server");
  EXPECT_TRUE(reply.version().empty());
}

TEST(InferenceServiceImpl, UnsupportedUnaryRpcsReturnUnimplemented)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, {at::kFloat});
  grpc::ServerContext ctx;

  using RpcInvoke = std::function<grpc::Status()>;
  const std::array<std::pair<const char*, RpcInvoke>, 11> test_cases = {{
      {"RepositoryIndex",
       [&service, &ctx]() {
         inference::RepositoryIndexRequest req;
         inference::RepositoryIndexResponse reply;
         return service.RepositoryIndex(&ctx, &req, &reply);
       }},
      {"RepositoryModelLoad",
       [&service, &ctx]() {
         inference::RepositoryModelLoadRequest req;
         inference::RepositoryModelLoadResponse reply;
         return service.RepositoryModelLoad(&ctx, &req, &reply);
       }},
      {"RepositoryModelUnload",
       [&service, &ctx]() {
         inference::RepositoryModelUnloadRequest req;
         inference::RepositoryModelUnloadResponse reply;
         return service.RepositoryModelUnload(&ctx, &req, &reply);
       }},
      {"SystemSharedMemoryStatus",
       [&service, &ctx]() {
         inference::SystemSharedMemoryStatusRequest req;
         inference::SystemSharedMemoryStatusResponse reply;
         return service.SystemSharedMemoryStatus(&ctx, &req, &reply);
       }},
      {"SystemSharedMemoryRegister",
       [&service, &ctx]() {
         inference::SystemSharedMemoryRegisterRequest req;
         inference::SystemSharedMemoryRegisterResponse reply;
         return service.SystemSharedMemoryRegister(&ctx, &req, &reply);
       }},
      {"SystemSharedMemoryUnregister",
       [&service, &ctx]() {
         inference::SystemSharedMemoryUnregisterRequest req;
         inference::SystemSharedMemoryUnregisterResponse reply;
         return service.SystemSharedMemoryUnregister(&ctx, &req, &reply);
       }},
      {"CudaSharedMemoryStatus",
       [&service, &ctx]() {
         inference::CudaSharedMemoryStatusRequest req;
         inference::CudaSharedMemoryStatusResponse reply;
         return service.CudaSharedMemoryStatus(&ctx, &req, &reply);
       }},
      {"CudaSharedMemoryRegister",
       [&service, &ctx]() {
         inference::CudaSharedMemoryRegisterRequest req;
         inference::CudaSharedMemoryRegisterResponse reply;
         return service.CudaSharedMemoryRegister(&ctx, &req, &reply);
       }},
      {"CudaSharedMemoryUnregister",
       [&service, &ctx]() {
         inference::CudaSharedMemoryUnregisterRequest req;
         inference::CudaSharedMemoryUnregisterResponse reply;
         return service.CudaSharedMemoryUnregister(&ctx, &req, &reply);
       }},
      {"TraceSetting",
       [&service, &ctx]() {
         inference::TraceSettingRequest req;
         inference::TraceSettingResponse reply;
         return service.TraceSetting(&ctx, &req, &reply);
       }},
      {"LogSettings",
       [&service, &ctx]() {
         inference::LogSettingsRequest req;
         inference::LogSettingsResponse reply;
         return service.LogSettings(&ctx, &req, &reply);
       }},
  }};

  for (const auto& [rpc_name, invoke] : test_cases) {
    SCOPED_TRACE(rpc_name);
    const auto status = invoke();
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNIMPLEMENTED);
    EXPECT_EQ(
        status.error_message(),
        std::string("RPC ") + rpc_name + " is not implemented");
  }
}

TEST(InferenceServiceImpl, ModelStreamInferReturnsUnimplemented)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, {at::kFloat});
  grpc::ServerContext ctx;

  auto status = service.ModelStreamInfer(&ctx, nullptr);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::UNIMPLEMENTED);
  EXPECT_EQ(status.error_message(), "RPC ModelStreamInfer is not implemented");
}

TEST(InferenceServiceImpl, ModelMetadataPopulatesInputsAndOutputs)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs = {
      torch::zeros({2, 2}, torch::TensorOptions().dtype(at::kFloat)),
      torch::zeros({3}, torch::TensorOptions().dtype(at::kLong))};
  std::vector<std::vector<int64_t>> input_dims = {{1, 2}, {3}};
  std::vector<std::string> input_names = {"input0", ""};
  std::vector<std::string> output_names = {"out0", ""};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, {at::kFloat, at::kLong},
      starpu_server::InferenceServiceImpl::InputShapeConfig{
          std::move(input_dims), 0},
      starpu_server::InferenceServiceImpl::ServiceOptions{
          .default_model_name = "server_model",
          .expected_input_names = std::move(input_names),
          .expected_output_names = std::move(output_names)});

  grpc::ServerContext ctx;
  inference::ModelMetadataRequest req;
  req.set_name("client_model");
  req.set_version("v1");
  inference::ModelMetadataResponse reply;

  auto status = service.ModelMetadata(&ctx, &req, &reply);

  ASSERT_TRUE(status.ok());
  EXPECT_EQ(reply.name(), "server_model");
  ASSERT_EQ(reply.versions_size(), 1);
  EXPECT_EQ(reply.versions(0), "v1");

  ASSERT_EQ(reply.inputs_size(), 2);
  EXPECT_EQ(reply.inputs(0).name(), "input0");
  EXPECT_EQ(reply.inputs(0).datatype(), "FP32");
  ASSERT_EQ(reply.inputs(0).shape_size(), 2);
  EXPECT_EQ(reply.inputs(0).shape(0), 1);
  EXPECT_EQ(reply.inputs(0).shape(1), 2);
  EXPECT_EQ(reply.inputs(1).name(), "input1");
  EXPECT_EQ(reply.inputs(1).datatype(), "INT64");
  ASSERT_EQ(reply.inputs(1).shape_size(), 1);
  EXPECT_EQ(reply.inputs(1).shape(0), 3);

  ASSERT_EQ(reply.outputs_size(), 2);
  EXPECT_EQ(reply.outputs(0).name(), "out0");
  EXPECT_EQ(reply.outputs(0).datatype(), "FP32");
  ASSERT_EQ(reply.outputs(0).shape_size(), 2);
  EXPECT_EQ(reply.outputs(0).shape(0), 2);
  EXPECT_EQ(reply.outputs(0).shape(1), 2);
  EXPECT_EQ(reply.outputs(1).name(), "output1");
  EXPECT_EQ(reply.outputs(1).datatype(), "INT64");
  ASSERT_EQ(reply.outputs(1).shape_size(), 1);
  EXPECT_EQ(reply.outputs(1).shape(0), 3);
}

TEST_F(InferenceServiceTest, ModelMetadataUsesRequestNameWhenNoDefaultModel)
{
  grpc::ServerContext ctx;
  inference::ModelMetadataRequest req;
  req.set_name("client_model");
  inference::ModelMetadataResponse reply;

  auto status = service->ModelMetadata(&ctx, &req, &reply);

  ASSERT_TRUE(status.ok());
  EXPECT_EQ(reply.name(), "client_model");
  EXPECT_EQ(reply.versions_size(), 0);
  ASSERT_EQ(reply.inputs_size(), 1);
  EXPECT_EQ(reply.inputs(0).name(), "input0");
  EXPECT_EQ(reply.inputs(0).datatype(), "FP32");
  EXPECT_EQ(reply.outputs_size(), 0);
}

TEST(InferenceServiceImpl, ConstructorRejectsUnsupportedInputDatatype)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  EXPECT_THROW(
      (void)starpu_server::InferenceServiceImpl(
          &queue, &ref_outputs, {at::kComplexFloat},
          starpu_server::InferenceServiceImpl::ServiceOptions{
              .default_model_name = "server_model"}),
      std::invalid_argument);
}

TEST(InferenceServiceImpl, ConstructorRejectsUnsupportedOutputDatatype)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs = {
      torch::zeros({1}, torch::TensorOptions().dtype(at::kComplexFloat))};
  EXPECT_THROW(
      (void)starpu_server::InferenceServiceImpl(
          &queue, &ref_outputs, {at::kFloat},
          starpu_server::InferenceServiceImpl::ServiceOptions{
              .default_model_name = "server_model"}),
      std::invalid_argument);
}

TEST(InferenceServiceImpl, ModelConfigPopulatesConfig)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs = {
      torch::zeros({2, 2}, torch::TensorOptions().dtype(at::kFloat)),
      torch::zeros({3}, torch::TensorOptions().dtype(at::kLong))};
  std::vector<std::vector<int64_t>> input_dims = {{1, 2}, {3}};
  std::vector<std::string> input_names = {"first", ""};
  std::vector<std::string> output_names = {"out0", ""};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, {at::kFloat, at::kLong},
      starpu_server::InferenceServiceImpl::InputShapeConfig{
          std::move(input_dims), 8},
      starpu_server::InferenceServiceImpl::ServiceOptions{
          .default_model_name = "server_model",
          .expected_input_names = std::move(input_names),
          .expected_output_names = std::move(output_names)});

  grpc::ServerContext ctx;
  inference::ModelConfigRequest req;
  req.set_name("client_model");
  inference::ModelConfigResponse reply;

  auto status = service.ModelConfig(&ctx, &req, &reply);

  ASSERT_TRUE(status.ok());
  const auto& config = reply.config();
  EXPECT_EQ(config.name(), "server_model");
  EXPECT_EQ(config.max_batch_size(), 8);

  ASSERT_EQ(config.input_size(), 2);
  EXPECT_EQ(config.input(0).name(), "first");
  EXPECT_EQ(config.input(0).data_type(), inference::DataType::TYPE_FP32);
  ASSERT_EQ(config.input(0).dims_size(), 2);
  EXPECT_EQ(config.input(0).dims(0), 1);
  EXPECT_EQ(config.input(0).dims(1), 2);
  EXPECT_EQ(config.input(1).name(), "input1");
  EXPECT_EQ(config.input(1).data_type(), inference::DataType::TYPE_INT64);
  ASSERT_EQ(config.input(1).dims_size(), 1);
  EXPECT_EQ(config.input(1).dims(0), 3);

  ASSERT_EQ(config.output_size(), 2);
  EXPECT_EQ(config.output(0).name(), "out0");
  EXPECT_EQ(config.output(0).data_type(), inference::DataType::TYPE_FP32);
  ASSERT_EQ(config.output(0).dims_size(), 2);
  EXPECT_EQ(config.output(0).dims(0), 2);
  EXPECT_EQ(config.output(0).dims(1), 2);
  EXPECT_EQ(config.output(1).name(), "output1");
  EXPECT_EQ(config.output(1).data_type(), inference::DataType::TYPE_INT64);
  ASSERT_EQ(config.output(1).dims_size(), 1);
  EXPECT_EQ(config.output(1).dims(0), 3);
}

TEST_F(InferenceServiceTest, ModelConfigUsesRequestNameWhenNoDefaultModel)
{
  grpc::ServerContext ctx;
  inference::ModelConfigRequest req;
  req.set_name("client_model");
  inference::ModelConfigResponse reply;

  auto status = service->ModelConfig(&ctx, &req, &reply);

  ASSERT_TRUE(status.ok());
  const auto& config = reply.config();
  EXPECT_EQ(config.name(), "client_model");
  EXPECT_EQ(config.max_batch_size(), 0);
  ASSERT_EQ(config.input_size(), 1);
  EXPECT_EQ(config.input(0).name(), "input0");
  EXPECT_EQ(config.input(0).data_type(), inference::DataType::TYPE_FP32);
  EXPECT_EQ(config.output_size(), 0);
}

TEST(InferenceServiceImpl, ConstructorRejectsUnsupportedInputDatatypeForConfig)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  EXPECT_THROW(
      (void)starpu_server::InferenceServiceImpl(
          &queue, &ref_outputs, {at::kComplexFloat}),
      std::invalid_argument);
}

TEST(InferenceServiceImpl, ConstructorRejectsUnsupportedOutputDatatypeForConfig)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs = {
      torch::zeros({1}, torch::TensorOptions().dtype(at::kComplexFloat))};
  EXPECT_THROW(
      (void)starpu_server::InferenceServiceImpl(
          &queue, &ref_outputs, {at::kFloat}),
      std::invalid_argument);
}

TEST(InferenceServiceImpl, ModelStatisticsRejectsNullRequest)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, {at::kFloat});

  grpc::ServerContext ctx;
  inference::ModelStatisticsResponse reply;

  auto status = service.ModelStatistics(&ctx, nullptr, &reply);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(), "Invalid request");
}

TEST(InferenceServiceImpl, ModelStatisticsRejectsNullReply)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, {at::kFloat});

  grpc::ServerContext ctx;
  inference::ModelStatisticsRequest req;

  auto status = service.ModelStatistics(&ctx, &req, nullptr);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(), "Invalid request");
}

TEST(InferenceServiceImpl, ModelStatisticsSkipsMismatchedName)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, {at::kFloat});
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown{};

  auto request_a = starpu_server::make_model_request("model_a", "v1");
  auto request_b = starpu_server::make_model_request("model_b", "v1");
  starpu_server::InferenceServiceImpl::TestAccessor::RecordSuccessForTest(
      &service, &request_a, breakdown, starpu_server::MonotonicClock::now(),
      "model_a");
  starpu_server::InferenceServiceImpl::TestAccessor::RecordSuccessForTest(
      &service, &request_b, breakdown, starpu_server::MonotonicClock::now(),
      "model_b");

  grpc::ServerContext ctx;
  inference::ModelStatisticsRequest req;
  req.set_name("model_a");
  inference::ModelStatisticsResponse reply;

  auto status = service.ModelStatistics(&ctx, &req, &reply);

  ASSERT_TRUE(status.ok());
  ASSERT_EQ(reply.model_stats_size(), 1);
  EXPECT_EQ(reply.model_stats(0).name(), "model_a");
}

TEST(InferenceServiceImpl, ModelStatisticsSkipsMismatchedVersion)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, {at::kFloat});
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown{};

  auto request_v1 = starpu_server::make_model_request("model_a", "v1");
  auto request_v2 = starpu_server::make_model_request("model_a", "v2");
  starpu_server::InferenceServiceImpl::TestAccessor::RecordSuccessForTest(
      &service, &request_v1, breakdown, starpu_server::MonotonicClock::now(),
      "model_a");
  starpu_server::InferenceServiceImpl::TestAccessor::RecordSuccessForTest(
      &service, &request_v2, breakdown, starpu_server::MonotonicClock::now(),
      "model_a");

  grpc::ServerContext ctx;
  inference::ModelStatisticsRequest req;
  req.set_name("model_a");
  req.set_version("v1");
  inference::ModelStatisticsResponse reply;

  auto status = service.ModelStatistics(&ctx, &req, &reply);

  ASSERT_TRUE(status.ok());
  ASSERT_EQ(reply.model_stats_size(), 1);
  EXPECT_EQ(reply.model_stats(0).version(), "v1");
}

TEST(InferenceServiceImpl, ModelStatisticsHandlesNullStatisticTarget)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, {at::kFloat});
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown{};
  auto request = starpu_server::make_model_request("model_a", "v1");
  starpu_server::InferenceServiceImpl::TestAccessor::RecordSuccessForTest(
      &service, &request, breakdown, starpu_server::MonotonicClock::now(),
      "model_a");

  ModelStatisticsNullTargetGuard guard;
  grpc::ServerContext ctx;
  inference::ModelStatisticsRequest req;
  req.set_name("model_a");
  inference::ModelStatisticsResponse reply;

  auto status = service.ModelStatistics(&ctx, &req, &reply);

  ASSERT_TRUE(status.ok());
  EXPECT_EQ(reply.model_stats_size(), 1);
}

TEST(InferenceServiceImpl, RequestBatchSizeHandlesEmptyShape)
{
  inference::ModelInferRequest req;
  auto* input = req.add_inputs();
  input->set_datatype("FP32");

  EXPECT_EQ(
      starpu_server::InferenceServiceImpl::TestAccessor::
          RequestBatchSizeForTest(&req, 4),
      1U);
}

TEST(InferenceServiceImpl, RequestBatchSizeHandlesNonPositiveBatch)
{
  inference::ModelInferRequest req;
  auto* input = req.add_inputs();
  input->set_datatype("FP32");
  input->add_shape(0);

  EXPECT_EQ(
      starpu_server::InferenceServiceImpl::TestAccessor::
          RequestBatchSizeForTest(&req, 4),
      1U);
}

TEST(InferenceServiceImpl, RequestBatchSizeReturnsBatch)
{
  inference::ModelInferRequest req;
  auto* input = req.add_inputs();
  input->set_datatype("FP32");
  input->add_shape(3);

  EXPECT_EQ(
      starpu_server::InferenceServiceImpl::TestAccessor::
          RequestBatchSizeForTest(&req, 4),
      3U);
}

TEST(InferenceServiceImpl, DurationMsToNsSaturatesToMax)
{
  const double duration_ms =
      static_cast<double>(std::numeric_limits<uint64_t>::max());

  EXPECT_EQ(
      starpu_server::InferenceServiceImpl::TestAccessor::DurationMsToNsForTest(
          duration_ms),
      std::numeric_limits<uint64_t>::max());
}

TEST(InferenceServiceImpl, ElapsedSinceReturnsZeroForFutureStart)
{
  const auto future_start =
      starpu_server::MonotonicClock::now() + std::chrono::seconds(1);

  EXPECT_EQ(
      starpu_server::InferenceServiceImpl::TestAccessor::ElapsedSinceForTest(
          future_start),
      0U);
}

TEST_F(InferenceServiceTest, ValidateInputsSuccess)
{
  auto req = starpu_server::make_valid_request();
  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1U);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(inputs[0].scalar_type(), at::kFloat);
  EXPECT_FLOAT_EQ(inputs[0][0][0].item<float>(), kF1);
}

TEST_F(InferenceServiceTest, ValidateInputsCopiesRequestBuffer)
{
  constexpr size_t kElements = 1U << 10;
  std::vector<float> data(kElements, kF1);
  auto req = starpu_server::make_model_infer_request({
      {{static_cast<int64_t>(kElements)},
       at::kFloat,
       starpu_server::to_raw_data(data)},
  });

  std::vector<torch::Tensor> inputs;
  std::vector<std::shared_ptr<const void>> keep_alive;
  auto status = service->validate_and_convert_inputs(&req, inputs, &keep_alive);

  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1U);
  ASSERT_EQ(keep_alive.size(), 1U);
  EXPECT_EQ(inputs[0].data_ptr(), const_cast<void*>(keep_alive[0].get()));
  EXPECT_NE(
      inputs[0].data_ptr(), const_cast<void*>(static_cast<const void*>(
                                req.raw_input_contents(0).data())));
  EXPECT_EQ(
      inputs[0].nbytes(),
      static_cast<int64_t>(req.raw_input_contents(0).size()));
}

TEST_F(InferenceServiceTest, ValidateInputsKeepAliveSharesOwnedBuffer)
{
  std::vector<float> data = {kF1, kF2, kF3, kF4};
  auto req = starpu_server::make_model_infer_request({
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data)},
  });

  std::vector<torch::Tensor> inputs;
  std::vector<std::shared_ptr<const void>> keep_alive;
  auto status = service->validate_and_convert_inputs(&req, inputs, &keep_alive);

  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1U);
  ASSERT_EQ(keep_alive.size(), 1U);
  EXPECT_EQ(inputs[0].data_ptr(), const_cast<void*>(keep_alive[0].get()));
  EXPECT_NE(
      keep_alive[0].get(),
      static_cast<const void*>(req.raw_input_contents(0).data()));
}

TEST_F(InferenceServiceTest, ValidateInputsNonContiguous)
{
  auto base = torch::tensor({{kF1, kF2}, {kF3, kF4}});
  auto noncontig = base.transpose(0, 1);
  auto contig = noncontig.contiguous();
  std::span<const float> span{
      contig.data_ptr<float>(), static_cast<size_t>(contig.numel())};
  std::vector<float> data(span.begin(), span.end());
  auto req = starpu_server::make_model_infer_request({
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data)},
  });
  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1U);
  EXPECT_TRUE(inputs[0].is_contiguous());
  EXPECT_TRUE(torch::allclose(inputs[0], contig));
}

TEST_F(MultiTypeInferenceServiceTest, ValidateInputsMultipleDtypes)
{
  std::vector<float> data0 = {kF1, kF2, kF3, kF4};
  std::vector<int64_t> data1 = {kI10, kI20, kI30};
  auto req = starpu_server::make_model_infer_request({
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data0)},
      {{3}, at::kLong, starpu_server::to_raw_data(data1)},
  });
  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 2U);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(inputs[0].scalar_type(), at::kFloat);
  EXPECT_FLOAT_EQ(inputs[0][0][0].item<float>(), kF1);
  EXPECT_EQ(inputs[1].sizes(), (torch::IntArrayRef{3}));
  EXPECT_EQ(inputs[1].scalar_type(), at::kLong);
  EXPECT_EQ(inputs[1][0].item<int64_t>(), kI10);
}

TEST_F(MultiTypeInferenceServiceTest, ValidateInputsRejectsDatatypeMismatch)
{
  std::vector<int64_t> data0 = {kI10, kI20, kI30};
  std::vector<float> data1 = {kF1, kF2, kF3, kF4};
  auto req = starpu_server::make_model_infer_request({
      {{3}, at::kLong, starpu_server::to_raw_data(data0)},
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data1)},
  });

  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&req, inputs);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(), "Input tensor datatype mismatch");
}

TEST_F(MultiTypeInferenceServiceTest, ValidateInputsRejectsMixedNamedAndUnnamed)
{
  std::vector<float> data0 = {kF1, kF2, kF3, kF4};
  std::vector<int64_t> data1 = {kI10, kI20, kI30};
  auto req = starpu_server::make_model_infer_request({
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data0)},
      {{3}, at::kLong, starpu_server::to_raw_data(data1)},
  });
  req.mutable_inputs(1)->clear_name();

  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&req, inputs);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(
      status.error_message(),
      "All input tensors must include a name when using named inputs");
}

TEST_F(
    NamedInputInferenceServiceTest,
    ValidateInputsRejectsUnnamedWhenNamesExpected)
{
  std::vector<float> data0 = {kF1, kF2, kF3, kF4};
  std::vector<int64_t> data1 = {kI10, kI20, kI30};
  auto req = starpu_server::make_model_infer_request({
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data0)},
      {{3}, at::kLong, starpu_server::to_raw_data(data1)},
  });
  req.mutable_inputs(0)->clear_name();
  req.mutable_inputs(1)->clear_name();

  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&req, inputs);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(), "Input tensor names must be provided");
}

TEST_F(NamedInputInferenceServiceTest, ValidateInputsRejectsEmptyConfiguredName)
{
  std::vector<float> data0 = {kF1, kF2, kF3, kF4};
  std::vector<int64_t> data1 = {kI10, kI20, kI30};
  auto req = starpu_server::make_model_infer_request({
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data0)},
      {{3}, at::kLong, starpu_server::to_raw_data(data1)},
  });
  req.mutable_inputs(0)->set_name("first");
  req.mutable_inputs(1)->set_name("second");

  starpu_server::InferenceServiceImpl::TestAccessor::
      SetExpectedInputNamesForTest(service.get(), {"", "second"});

  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&req, inputs);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(), "Configured input name missing at index 0");
}

TEST_F(NamedInputInferenceServiceTest, ValidateInputsRejectsUnexpectedName)
{
  std::vector<float> data0 = {kF1, kF2, kF3, kF4};
  std::vector<int64_t> data1 = {kI10, kI20, kI30};
  auto req = starpu_server::make_model_infer_request({
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data0)},
      {{3}, at::kLong, starpu_server::to_raw_data(data1)},
  });
  req.mutable_inputs(0)->set_name("first");
  req.mutable_inputs(1)->set_name("unknown");

  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&req, inputs);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(), "Unexpected input tensor name 'unknown'");
}

TEST_F(
    NamedInputInferenceServiceTest, ValidateInputsRejectsDuplicateRequestName)
{
  std::vector<float> data0 = {kF1, kF2, kF3, kF4};
  std::vector<int64_t> data1 = {kI10, kI20, kI30};
  auto req = starpu_server::make_model_infer_request({
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data0)},
      {{3}, at::kLong, starpu_server::to_raw_data(data1)},
  });
  req.mutable_inputs(0)->set_name("first");
  req.mutable_inputs(1)->set_name("first");

  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&req, inputs);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(), "Input tensor name 'first' is duplicated");
}

TEST(InferenceServiceImpl, ValidateInputsRejectsDuplicateConfiguredNames)
{
  std::vector<float> data0 = {kF1, kF2, kF3, kF4};
  std::vector<int64_t> data1 = {kI10, kI20, kI30};
  auto req = starpu_server::make_model_infer_request({
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data0)},
      {{3}, at::kLong, starpu_server::to_raw_data(data1)},
  });

  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat, at::kLong};
  std::vector<std::string> expected_names = {"dup", "dup"};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, std::move(expected_types),
      starpu_server::InferenceServiceImpl::ServiceOptions{
          .expected_input_names = std::move(expected_names)});

  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(
      status.error_message(), "Configured input name 'dup' is duplicated");
}

TEST_F(NamedInputInferenceServiceTest, ValidateInputsReordersByNameMapping)
{
  std::vector<int64_t> data1 = {kI10, kI20, kI30};
  std::vector<float> data0 = {kF1, kF2, kF3, kF4};
  auto req = starpu_server::make_model_infer_request({
      {{3}, at::kLong, starpu_server::to_raw_data(data1)},
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data0)},
  });
  req.mutable_inputs(0)->set_name("second");
  req.mutable_inputs(1)->set_name("first");

  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&req, inputs);

  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 2U);
  EXPECT_EQ(inputs[0].scalar_type(), at::kFloat);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_FLOAT_EQ(inputs[0][0][0].item<float>(), kF1);
  EXPECT_EQ(inputs[1].scalar_type(), at::kLong);
  EXPECT_EQ(inputs[1].sizes(), (torch::IntArrayRef{3}));
  EXPECT_EQ(inputs[1][0].item<int64_t>(), kI10);
}

TEST_F(NamedInputInferenceServiceTest, ValidateInputsReordersKeepAliveByName)
{
  std::vector<int64_t> data1 = {kI10, kI20, kI30};
  std::vector<float> data0 = {kF1, kF2, kF3, kF4};
  auto req = starpu_server::make_model_infer_request({
      {{3}, at::kLong, starpu_server::to_raw_data(data1)},
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data0)},
  });
  req.mutable_inputs(0)->set_name("second");
  req.mutable_inputs(1)->set_name("first");

  std::vector<torch::Tensor> inputs;
  std::vector<std::shared_ptr<const void>> keep_alive;
  auto status = service->validate_and_convert_inputs(&req, inputs, &keep_alive);

  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 2U);
  ASSERT_EQ(keep_alive.size(), 2U);
  EXPECT_EQ(inputs[0].data_ptr(), const_cast<void*>(keep_alive[0].get()));
  EXPECT_EQ(inputs[1].data_ptr(), const_cast<void*>(keep_alive[1].get()));
  EXPECT_EQ(inputs[0].scalar_type(), at::kFloat);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_FLOAT_EQ(inputs[0][0][0].item<float>(), kF1);
  EXPECT_EQ(inputs[1].scalar_type(), at::kLong);
  EXPECT_EQ(inputs[1].sizes(), (torch::IntArrayRef{3}));
  EXPECT_EQ(inputs[1][0].item<int64_t>(), kI10);
}

TEST_F(
    ConfiguredShapeNoBatchingInferenceServiceTest,
    ValidateInputsConfiguredShapeNoBatching)
{
  auto valid_req = starpu_server::make_shape_request({2, 2});
  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&valid_req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1U);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));

  auto mismatched_dims_req = starpu_server::make_shape_request({2, 3});
  inputs.clear();
  status = service->validate_and_convert_inputs(&mismatched_dims_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  auto unexpected_rank_req = starpu_server::make_shape_request({2, 2, 2});
  inputs.clear();
  status = service->validate_and_convert_inputs(&unexpected_rank_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(
    ConfiguredShapeWithBatchingInferenceServiceTest,
    ValidateInputsConfiguredShapeWithBatching)
{
  auto same_rank_req = starpu_server::make_shape_request({2, 2});
  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&same_rank_req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1U);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));

  auto batch_only_req = starpu_server::make_shape_request({2});
  inputs.clear();
  status = service->validate_and_convert_inputs(&batch_only_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  auto explicit_batch_req = starpu_server::make_shape_request({3, 2, 2});
  inputs.clear();
  status = service->validate_and_convert_inputs(&explicit_batch_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  auto exceeding_batch_req = starpu_server::make_shape_request({5, 2, 2});
  inputs.clear();
  status = service->validate_and_convert_inputs(&exceeding_batch_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  auto mismatched_dims_req = starpu_server::make_shape_request({2, 3});
  inputs.clear();
  status = service->validate_and_convert_inputs(&mismatched_dims_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  auto mismatched_tail_req = starpu_server::make_shape_request({3, 2, 3});
  inputs.clear();
  status = service->validate_and_convert_inputs(&mismatched_tail_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(
    ConfiguredShapeWithBatchingInferenceServiceTest,
    ValidateInputsConfiguredShapeRejectsZeroBatchSize)
{
  auto zero_batch_override_req = starpu_server::make_shape_request({0, 2});
  std::vector<torch::Tensor> inputs;
  auto status =
      service->validate_and_convert_inputs(&zero_batch_override_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(
    ConfiguredShapeWithBatchingInferenceServiceTest,
    ValidateInputsConfiguredShapeAllowsBatchOverrideWhenRanksMatch)
{
  auto overridden_batch_req = starpu_server::make_shape_request({3, 2});
  std::vector<torch::Tensor> inputs;
  auto status =
      service->validate_and_convert_inputs(&overridden_batch_req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1U);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{3, 2}));
}

TEST_F(
    ConfiguredShapeWithBatchingInferenceServiceTest,
    ValidateInputsConfiguredShapeRejectsTailLengthMismatch)
{
  auto mismatched_tail_length_req =
      starpu_server::make_shape_request({3, 2, 2, 2});
  std::vector<torch::Tensor> inputs;
  auto status =
      service->validate_and_convert_inputs(&mismatched_tail_length_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(
    ZeroRankConfiguredShapeWithBatchingInferenceServiceTest,
    ValidateInputsConfiguredShapeRejectsZeroRankBatch)
{
  auto zero_rank_req = starpu_server::make_shape_request({});
  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&zero_rank_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(InferenceServiceTest, ValidateInputsMismatchedCount)
{
  std::vector<float> data0 = {kF1, kF2, kF3, kF4};
  std::vector<int64_t> data1 = {kI10, kI20, kI30};
  auto req = starpu_server::make_model_infer_request({
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data0)},
      {{3}, at::kLong, starpu_server::to_raw_data(data1)},
  });
  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(InferenceServiceTest, ValidateInputsRejectsNullRequest)
{
  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(nullptr, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(), "ModelInfer request is null");
}

TEST(InferenceServiceImpl, PopulateResponsePopulatesFieldsAndTimes)
{
  auto req = starpu_server::make_model_request("model", "1");
  std::vector<torch::Tensor> outputs = {
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kInt))};
  inference::ModelInferResponse reply;
  int64_t recv_ms = kI10;
  int64_t send_ms = kI20;
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;
  breakdown.preprocess_ms = 0.5;
  breakdown.queue_ms = 1.0;
  breakdown.batch_ms = 0.75;
  breakdown.submit_ms = 2.0;
  breakdown.scheduling_ms = 3.0;
  breakdown.codelet_ms = 4.0;
  breakdown.inference_ms = 5.0;
  breakdown.callback_ms = 6.0;
  breakdown.postprocess_ms = 0.5;
  breakdown.total_ms = 7.0;
  breakdown.overall_ms = 8.0;
  auto status = starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, outputs, recv_ms, breakdown);
  ASSERT_TRUE(status.ok());
  reply.set_server_send_ms(send_ms);
  starpu_server::verify_populate_response(
      req, reply, outputs, recv_ms, send_ms, breakdown);
}

TEST(InferenceServiceImpl, PopulateResponseRejectsNullRequest)
{
  inference::ModelInferResponse reply;
  std::vector<torch::Tensor> outputs = {
      torch::tensor({1}, torch::TensorOptions().dtype(at::kInt))};
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;

  auto status = starpu_server::InferenceServiceImpl::populate_response(
      nullptr, &reply, outputs, 0, breakdown);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(), "ModelInfer request is null");
}

TEST(InferenceServiceImpl, PopulateResponseRejectsNullReply)
{
  auto req = starpu_server::make_model_request("model", "1");
  std::vector<torch::Tensor> outputs = {
      torch::tensor({1}, torch::TensorOptions().dtype(at::kInt))};
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;

  auto status = starpu_server::InferenceServiceImpl::populate_response(
      &req, nullptr, outputs, 0, breakdown);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(), "ModelInfer response is null");
}

TEST(InferenceServiceImpl, PopulateResponseUsesOverrideModelName)
{
  auto req = starpu_server::make_model_request("client_model", "1");
  std::vector<torch::Tensor> outputs = {
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kInt))};
  inference::ModelInferResponse reply;
  int64_t recv_ms = kI10;
  int64_t send_ms = kI20;
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;
  breakdown.total_ms = 1.0;
  breakdown.overall_ms = 2.0;

  const std::string server_model = "server_model";
  starpu_server::InferenceServiceImpl::PopulateResponseOptions options;
  options.model_name_override = server_model;
  auto status = starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, outputs, recv_ms, breakdown, options);
  ASSERT_TRUE(status.ok());
  reply.set_server_send_ms(send_ms);
  starpu_server::verify_populate_response(
      req, reply, outputs, recv_ms, send_ms, breakdown, server_model);
}

TEST(InferenceServiceImpl, PopulateResponseRejectsEmptyRequestedOutputName)
{
  auto req = starpu_server::make_model_request("model", "1");
  req.add_outputs();
  std::vector<torch::Tensor> outputs = {
      torch::tensor({1, 2}, torch::TensorOptions().dtype(at::kInt))};
  inference::ModelInferResponse reply;
  int64_t recv_ms = 0;
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;
  std::vector<std::string> output_names = {"out0"};

  starpu_server::InferenceServiceImpl::PopulateResponseOptions options;
  options.output_names = output_names;
  auto status = starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, outputs, recv_ms, breakdown, options);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(
      status.error_message().find("Requested output name must be non-empty"),
      std::string::npos);
}

TEST(InferenceServiceImpl, PopulateResponseRejectsUnknownRequestedOutputName)
{
  auto req = starpu_server::make_model_request("model", "1");
  req.add_outputs()->set_name("missing");
  std::vector<torch::Tensor> outputs = {
      torch::tensor({1, 2}, torch::TensorOptions().dtype(at::kInt))};
  inference::ModelInferResponse reply;
  int64_t recv_ms = 0;
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;
  std::vector<std::string> output_names = {"known"};

  starpu_server::InferenceServiceImpl::PopulateResponseOptions options;
  options.output_names = output_names;
  auto status = starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, outputs, recv_ms, breakdown, options);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(
      status.error_message().find(
          "Requested output 'missing' is not available"),
      std::string::npos);
}

TEST(InferenceServiceImpl, PopulateResponseRejectsDuplicateRequestedOutputName)
{
  auto req = starpu_server::make_model_request("model", "1");
  req.add_outputs()->set_name("dup");
  req.add_outputs()->set_name("dup");
  std::vector<torch::Tensor> outputs = {
      torch::tensor({1, 2}, torch::TensorOptions().dtype(at::kInt))};
  inference::ModelInferResponse reply;
  int64_t recv_ms = 0;
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;
  std::vector<std::string> output_names = {"dup"};

  starpu_server::InferenceServiceImpl::PopulateResponseOptions options;
  options.output_names = output_names;
  auto status = starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, outputs, recv_ms, breakdown, options);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(
      status.error_message().find("Requested output 'dup' is duplicated"),
      std::string::npos);
}

TEST(
    InferenceServiceImpl, PopulateResponseRejectsDuplicateConfiguredOutputNames)
{
  auto req = starpu_server::make_model_request("model", "1");
  req.add_outputs()->set_name("dup");
  std::vector<torch::Tensor> outputs = {
      torch::tensor({1, 2}, torch::TensorOptions().dtype(at::kInt)),
      torch::tensor({3, 4}, torch::TensorOptions().dtype(at::kInt))};
  inference::ModelInferResponse reply;
  int64_t recv_ms = 0;
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;
  std::vector<std::string> output_names = {"dup", "dup"};

  starpu_server::InferenceServiceImpl::PopulateResponseOptions options;
  options.output_names = output_names;
  auto status = starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, outputs, recv_ms, breakdown, options);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(
      status.error_message().find("Configured output name 'dup' is duplicated"),
      std::string::npos);
}

TEST(InferenceServiceImpl, PopulateResponseHandlesNonContiguousOutputs)
{
  auto req = starpu_server::make_model_request("model", "1");
  auto base = torch::tensor({{1, 2}, {3, 4}});
  auto noncontig = base.transpose(0, 1);
  ASSERT_FALSE(noncontig.is_contiguous());
  std::vector<torch::Tensor> outputs = {noncontig};
  inference::ModelInferResponse reply;
  int64_t recv_ms = kI10;
  int64_t send_ms = kI20;
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;
  breakdown.preprocess_ms = 0.25;
  breakdown.queue_ms = 0.5;
  breakdown.batch_ms = 0.4;
  breakdown.submit_ms = 1.5;
  breakdown.scheduling_ms = 2.5;
  breakdown.codelet_ms = 3.5;
  breakdown.inference_ms = 4.5;
  breakdown.callback_ms = 5.5;
  breakdown.postprocess_ms = 0.25;
  breakdown.total_ms = 6.5;
  breakdown.overall_ms = 7.0;
  auto status = starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, outputs, recv_ms, breakdown);
  ASSERT_TRUE(status.ok());
  auto contig = noncontig.contiguous();
  reply.set_server_send_ms(send_ms);
  starpu_server::verify_populate_response(
      req, reply, {contig}, recv_ms, send_ms, breakdown);
}

TEST(InferenceServiceImpl, PopulateResponseHandlesCudaOutputs)
{
  skip_if_no_cuda();

  auto req = starpu_server::make_model_request("model", "1");
  auto options =
      torch::TensorOptions().dtype(at::kFloat).device(torch::kCUDA, 0);
  auto gpu_tensor = torch::arange(0, 6, options).view({2, 3});
  auto cpu_tensor = gpu_tensor.to(torch::kCPU);
  std::vector<torch::Tensor> outputs = {gpu_tensor};
  inference::ModelInferResponse reply;
  int64_t recv_ms = kI10;
  int64_t send_ms = kI20;
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;
  breakdown.preprocess_ms = 0.75;
  breakdown.queue_ms = 1.25;
  breakdown.batch_ms = 1.0;
  breakdown.submit_ms = 2.25;
  breakdown.scheduling_ms = 3.25;
  breakdown.codelet_ms = 4.25;
  breakdown.inference_ms = 5.25;
  breakdown.callback_ms = 6.25;
  breakdown.postprocess_ms = 0.75;
  breakdown.total_ms = 7.25;
  breakdown.overall_ms = 8.25;

  auto status = starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, outputs, recv_ms, breakdown);

  ASSERT_TRUE(status.ok());
  ASSERT_EQ(reply.outputs_size(), 1);
  ASSERT_EQ(reply.raw_output_contents_size(), 1);
  const auto& out_meta = reply.outputs(0);
  EXPECT_EQ(out_meta.shape_size(), cpu_tensor.dim());
  for (int64_t idx = 0; idx < cpu_tensor.dim(); ++idx) {
    EXPECT_EQ(out_meta.shape(idx), cpu_tensor.size(idx));
  }
  EXPECT_EQ(
      out_meta.datatype(),
      starpu_server::scalar_type_to_datatype(cpu_tensor.scalar_type()));

  reply.set_server_send_ms(send_ms);
  starpu_server::verify_populate_response(
      req, reply, {cpu_tensor}, recv_ms, send_ms, breakdown);
}

TEST(InferenceServiceImpl, PopulateResponseDetectsOverflow)
{
  auto req = starpu_server::make_model_request("model", "1");
  static float dummy;
  const size_t huge_elems =
      std::numeric_limits<size_t>::max() / sizeof(float) + 1;
  auto huge_tensor = torch::from_blob(
      &dummy, {static_cast<int64_t>(huge_elems)},
      torch::TensorOptions().dtype(at::kFloat));
  inference::ModelInferResponse reply;
  int64_t recv_ms = 0;
  int64_t send_ms = 0;
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;
  auto status = starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, {huge_tensor}, recv_ms, breakdown);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceServiceImpl, NormalizeNamesReturnsEmptyWhenAllUnnamed)
{
  auto names =
      starpu_server::InferenceServiceImpl::TestAccessor::NormalizeNamesForTest(
          std::vector<std::string>{"", ""}, 2, "input", "input");
  EXPECT_TRUE(names.empty());
}

TEST(InferenceServiceImpl, NormalizeNamesReturnsEmptyOnSizeMismatch)
{
  auto names =
      starpu_server::InferenceServiceImpl::TestAccessor::NormalizeNamesForTest(
          std::vector<std::string>{"input0"}, 2, "input", "input");
  EXPECT_TRUE(names.empty());
}

TEST(InferenceServiceImpl, NormalizeNamesReturnsEmptyWhenExpectedSizeZero)
{
  auto names =
      starpu_server::InferenceServiceImpl::TestAccessor::NormalizeNamesForTest(
          std::vector<std::string>{"input0"}, 0, "input", "input");
  EXPECT_TRUE(names.empty());
}

TEST(InferenceServiceImpl, NormalizeNamesFillsMissingEntries)
{
  auto names =
      starpu_server::InferenceServiceImpl::TestAccessor::NormalizeNamesForTest(
          std::vector<std::string>{"", "out1"}, 2, "output", "output");
  ASSERT_EQ(names.size(), 2U);
  EXPECT_EQ(names[0], "output0");
  EXPECT_EQ(names[1], "out1");
}

TEST(InferenceServiceImpl, MissingNamedInputReportsError)
{
  std::vector<bool> filled = {true, false};
  std::vector<std::string> names = {"first", "second"};

  auto status = starpu_server::InferenceServiceImpl::TestAccessor::
      CheckMissingInputsForTest(filled, std::span<const std::string>(names));

  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(), "Missing input tensor 'second'");
}

TEST(InferenceServiceImpl, RpcDoneTagArmHandlesNullContext)
{
  EXPECT_NO_THROW(starpu_server::InferenceServiceImpl::TestAccessor::
                      ArmRpcDoneTagWithNullContextForTest());
}

TEST(InferenceServiceImpl, GrpcHealthStatusHandlesNullServer)
{
  EXPECT_NO_THROW(starpu_server::InferenceServiceImpl::TestAccessor::
                      SetGrpcHealthStatusForTest(nullptr, true));
}

TEST_F(InferenceServiceTest, RecordSuccessHandlesNullRequest)
{
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown{};
  EXPECT_NO_THROW(
      starpu_server::InferenceServiceImpl::TestAccessor::RecordSuccessForTest(
          service.get(), nullptr, breakdown,
          starpu_server::MonotonicClock::now(), "model"));
}

TEST_F(InferenceServiceTest, RecordFailureHandlesNullRequest)
{
  EXPECT_NO_THROW(
      starpu_server::InferenceServiceImpl::TestAccessor::RecordFailureForTest(
          service.get(), nullptr, starpu_server::MonotonicClock::now(),
          "model"));
}

TEST(InferenceServiceImpl, IsContextCancelledHandlesNullContext)
{
  EXPECT_FALSE(starpu_server::InferenceServiceImpl::TestAccessor::
                   IsContextCancelledForTest(nullptr));
}

TEST(InferenceServiceImpl, ScalarTypeToModelDtypeMapsKnownTypes)
{
  using DataType = inference::DataType;
  struct Case {
    at::ScalarType type;
    DataType expected;
  };
  const std::array<Case, 10> cases = {{
      {at::kBool, DataType::TYPE_BOOL},
      {at::kByte, DataType::TYPE_UINT8},
      {at::kChar, DataType::TYPE_INT8},
      {at::kShort, DataType::TYPE_INT16},
      {at::kInt, DataType::TYPE_INT32},
      {at::kLong, DataType::TYPE_INT64},
      {at::kHalf, DataType::TYPE_FP16},
      {at::kFloat, DataType::TYPE_FP32},
      {at::kDouble, DataType::TYPE_FP64},
      {at::kBFloat16, DataType::TYPE_BF16},
  }};

  for (const auto& test_case : cases) {
    EXPECT_EQ(
        starpu_server::InferenceServiceImpl::TestAccessor::
            ScalarTypeToModelDtypeForTest(test_case.type),
        test_case.expected);
  }
}

TEST(InferenceServiceImpl, ScalarTypeToModelDtypeReturnsInvalidForUnsupported)
{
  EXPECT_EQ(
      starpu_server::InferenceServiceImpl::TestAccessor::
          ScalarTypeToModelDtypeForTest(at::kComplexFloat),
      inference::DataType::TYPE_INVALID);
}

TEST(InferenceServiceImpl, ResolveTensorNameUsesExplicitName)
{
  std::vector<std::string> names = {"first", "second"};
  EXPECT_EQ(
      starpu_server::InferenceServiceImpl::TestAccessor::
          ResolveTensorNameForTest(
              1, std::span<const std::string>(names), "input"),
      "second");
}

TEST(InferenceServiceImpl, ResolveTensorNameFallsBackForEmptyEntry)
{
  std::vector<std::string> names = {"", "second"};
  EXPECT_EQ(
      starpu_server::InferenceServiceImpl::TestAccessor::
          ResolveTensorNameForTest(
              0, std::span<const std::string>(names), "input"),
      "input0");
}

TEST(InferenceServiceImpl, ResolveTensorNameFallsBackForOutOfRange)
{
  std::vector<std::string> names = {"first"};
  EXPECT_EQ(
      starpu_server::InferenceServiceImpl::TestAccessor::
          ResolveTensorNameForTest(
              2, std::span<const std::string>(names), "output"),
      "output2");
}

TEST(InferenceServiceImpl, RpcDoneTagProceedInvokesOnDoneWhenOk)
{
  const bool called = starpu_server::InferenceServiceImpl::TestAccessor::
      RpcDoneTagProceedForTest(true, true);
  EXPECT_TRUE(called);
}

TEST(InferenceServiceImpl, RpcDoneTagProceedSkipsOnDoneWhenNotOk)
{
  const bool called = starpu_server::InferenceServiceImpl::TestAccessor::
      RpcDoneTagProceedForTest(false, true);
  EXPECT_FALSE(called);
}

TEST(InferenceServiceImpl, RpcDoneTagProceedSkipsWhenNoOnDone)
{
  const bool called = starpu_server::InferenceServiceImpl::TestAccessor::
      RpcDoneTagProceedForTest(true, false);
  EXPECT_FALSE(called);
}

TEST(InferenceServiceImpl, FillOutputTensorRejectsOutOfRangeIndex)
{
  inference::ModelInferResponse reply;
  std::vector<torch::Tensor> outputs = {
      torch::tensor({1.0F}, torch::TensorOptions().dtype(at::kFloat))};
  std::vector<std::size_t> output_indices = {1U};
  std::vector<std::string> output_names = {"out0"};

  auto status = starpu_server::InferenceServiceImpl::TestAccessor::
      FillOutputTensorForTest(&reply, outputs, output_indices, output_names);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(reply.outputs_size(), 0);
  EXPECT_EQ(reply.raw_output_contents_size(), 0);
}

TEST(InferenceServiceImpl, FillOutputTensorRejectsNullReply)
{
  std::vector<torch::Tensor> outputs = {
      torch::tensor({1.0F}, torch::TensorOptions().dtype(at::kFloat))};
  std::vector<std::size_t> output_indices = {0U};
  std::vector<std::string> output_names = {"out0"};

  auto status = starpu_server::InferenceServiceImpl::TestAccessor::
      FillOutputTensorForTest(nullptr, outputs, output_indices, output_names);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(), "ModelInfer response is null");
}

TEST(InferenceServiceImpl, FillOutputTensorUsesFallbackName)
{
  inference::ModelInferResponse reply;
  std::vector<torch::Tensor> outputs = {
      torch::tensor({1.0F, 2.0F}, torch::TensorOptions().dtype(at::kFloat)),
      torch::tensor({3.0F, 4.0F}, torch::TensorOptions().dtype(at::kFloat))};
  std::vector<std::size_t> output_indices = {1U};
  std::vector<std::string> output_names = {"named0"};

  auto status = starpu_server::InferenceServiceImpl::TestAccessor::
      FillOutputTensorForTest(&reply, outputs, output_indices, output_names);

  ASSERT_TRUE(status.ok());
  ASSERT_EQ(reply.outputs_size(), 1);
  EXPECT_EQ(reply.outputs(0).name(), "output1");
}

TEST(InferenceServiceImpl, BuildLatencyBreakdownMapsTimingFields)
{
  const auto base = starpu_server::MonotonicClock::time_point{};
  starpu_server::detail::TimingInfo info{};
  info.enqueued_time = base;
  info.dequeued_time = base + std::chrono::milliseconds(10);
  info.batch_collect_start_time = base + std::chrono::milliseconds(12);
  info.batch_collect_end_time = base + std::chrono::milliseconds(20);
  info.before_starpu_submitted_time = base + std::chrono::milliseconds(30);
  info.codelet_start_time = base + std::chrono::milliseconds(45);
  info.codelet_end_time = base + std::chrono::milliseconds(60);
  info.inference_start_time = base + std::chrono::milliseconds(70);
  info.callback_start_time = base + std::chrono::milliseconds(85);
  info.callback_end_time = base + std::chrono::milliseconds(100);

  auto breakdown = starpu_server::InferenceServiceImpl::TestAccessor::
      BuildLatencyBreakdownForTest(info, 123.0);

  EXPECT_DOUBLE_EQ(breakdown.queue_ms, 10.0);
  EXPECT_DOUBLE_EQ(breakdown.batch_ms, 8.0);
  EXPECT_DOUBLE_EQ(breakdown.submit_ms, 18.0);
  EXPECT_DOUBLE_EQ(breakdown.scheduling_ms, 15.0);
  EXPECT_DOUBLE_EQ(breakdown.codelet_ms, 15.0);
  EXPECT_DOUBLE_EQ(breakdown.inference_ms, 15.0);
  EXPECT_DOUBLE_EQ(breakdown.callback_ms, 15.0);
  EXPECT_DOUBLE_EQ(breakdown.total_ms, 123.0);
  EXPECT_DOUBLE_EQ(breakdown.preprocess_ms, 0.0);
  EXPECT_DOUBLE_EQ(breakdown.postprocess_ms, 0.0);
  EXPECT_DOUBLE_EQ(breakdown.overall_ms, 0.0);
}

TEST(InferenceServiceImpl, BuildLatencyBreakdownUsesDequeuedFallback)
{
  const auto base = starpu_server::MonotonicClock::time_point{};
  starpu_server::detail::TimingInfo info{};
  info.enqueued_time = base + std::chrono::milliseconds(5);
  info.dequeued_time = base + std::chrono::milliseconds(15);
  info.batch_collect_start_time = starpu_server::MonotonicClock::time_point{};
  info.batch_collect_end_time = starpu_server::MonotonicClock::time_point{};
  info.before_starpu_submitted_time = base + std::chrono::milliseconds(25);
  info.codelet_start_time = base + std::chrono::milliseconds(35);
  info.codelet_end_time = base + std::chrono::milliseconds(45);
  info.inference_start_time = base + std::chrono::milliseconds(55);
  info.callback_start_time = base + std::chrono::milliseconds(70);
  info.callback_end_time = base + std::chrono::milliseconds(80);

  auto breakdown = starpu_server::InferenceServiceImpl::TestAccessor::
      BuildLatencyBreakdownForTest(info, 80.5);

  EXPECT_DOUBLE_EQ(breakdown.queue_ms, 10.0);
  EXPECT_DOUBLE_EQ(breakdown.batch_ms, 0.0);
  EXPECT_DOUBLE_EQ(breakdown.submit_ms, 10.0);
  EXPECT_DOUBLE_EQ(breakdown.scheduling_ms, 10.0);
  EXPECT_DOUBLE_EQ(breakdown.codelet_ms, 10.0);
  EXPECT_DOUBLE_EQ(breakdown.inference_ms, 15.0);
  EXPECT_DOUBLE_EQ(breakdown.callback_ms, 10.0);
  EXPECT_DOUBLE_EQ(breakdown.total_ms, 80.5);
}

TEST(
    InferenceServiceImpl,
    HandleAsyncInferCompletionReturnsWhenCancelledInitially)
{
  starpu_server::InferenceServiceImpl::TestAccessor::
      ClearHandleAsyncInferCompletionTestHooks();
  const bool called = starpu_server::InferenceServiceImpl::TestAccessor::
      HandleAsyncInferCompletionForTest(true);
  EXPECT_FALSE(called);
}

TEST(
    InferenceServiceImpl,
    HandleAsyncInferCompletionReturnsWhenCancelledAfterTryAcquire)
{
  starpu_server::InferenceServiceImpl::HandleAsyncInferCompletionTestHooks
      hooks;
  hooks.after_try_acquire = [](const std::shared_ptr<std::atomic<bool>>& flag) {
    flag->store(true, std::memory_order_release);
  };
  HandleAsyncInferCompletionHooksGuard guard{std::move(hooks)};
  const bool called = starpu_server::InferenceServiceImpl::TestAccessor::
      HandleAsyncInferCompletionForTest(false);
  EXPECT_FALSE(called);
}

TEST(
    InferenceServiceImpl,
    HandleAsyncInferCompletionReturnsWhenCancelledBeforeFinalCallback)
{
  starpu_server::InferenceServiceImpl::HandleAsyncInferCompletionTestHooks
      hooks;
  hooks.before_final_cancel_check =
      [](const std::shared_ptr<std::atomic<bool>>& flag) {
        flag->store(true, std::memory_order_release);
      };
  HandleAsyncInferCompletionHooksGuard guard{std::move(hooks)};
  const bool called = starpu_server::InferenceServiceImpl::TestAccessor::
      HandleAsyncInferCompletionForTest(false);
  EXPECT_FALSE(called);
}

TEST(InferenceServiceImpl, ModelReadyRejectsMismatchedName)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, {at::kFloat},
      starpu_server::InferenceServiceImpl::ServiceOptions{
          .default_model_name = "default-model"});
  grpc::ServerContext context;
  inference::ModelReadyRequest request;
  inference::ModelReadyResponse response;

  request.set_name("other-model");

  auto status = service.ModelReady(&context, &request, &response);
  ASSERT_TRUE(status.ok());
  EXPECT_FALSE(response.ready());
}

TEST_F(InferenceServiceTest, HandleModelInferAsyncWorksWithNullContext)
{
  auto request = starpu_server::make_valid_request();
  auto expected_outputs = std::vector<torch::Tensor>{
      torch::tensor({kF2}, torch::TensorOptions().dtype(at::kFloat))};
  auto worker = prepare_job(expected_outputs, expected_outputs);

  std::promise<grpc::Status> status_promise;
  auto status_future = status_promise.get_future();

  service->HandleModelInferAsync(
      nullptr, &request, &reply, [&status_promise](grpc::Status status) {
        status_promise.set_value(std::move(status));
      });

  grpc::Status status = status_future.get();
  worker.join();

  EXPECT_TRUE(status.ok());
  ASSERT_EQ(reply.outputs_size(), 1);
}

TEST_F(InferenceServiceTest, HandleModelInferAsyncRejectsNullRequest)
{
  std::promise<grpc::Status> status_promise;
  auto status_future = status_promise.get_future();

  service->HandleModelInferAsync(
      &ctx, nullptr, &reply, [&status_promise](grpc::Status status) {
        status_promise.set_value(std::move(status));
      });

  const grpc::Status status = status_future.get();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(), "ModelInfer request is null");
  expect_empty_infer_response(reply);
}

TEST_F(InferenceServiceTest, HandleModelInferAsyncRejectsNullReply)
{
  auto request = starpu_server::make_valid_request();
  std::promise<grpc::Status> status_promise;
  auto status_future = status_promise.get_future();

  service->HandleModelInferAsync(
      &ctx, &request, nullptr, [&status_promise](grpc::Status status) {
        status_promise.set_value(std::move(status));
      });

  const grpc::Status status = status_future.get();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(), "ModelInfer response is null");
}

TEST_F(
    InferenceServiceTest, HandleModelInferAsyncSkipsErrorCallbackWhenCancelled)
{
  auto request = starpu_server::make_valid_request();
  request.add_raw_input_contents("extra");
  std::atomic<bool> callback_called{false};

  starpu_server::InferenceServiceImpl::HandleModelInferAsyncTestHooks hooks;
  hooks.on_cancel_flag_created =
      [](const std::shared_ptr<std::atomic<bool>>& cancel_flag) {
        cancel_flag->store(true, std::memory_order_release);
      };
  HandleModelInferAsyncHooksGuard guard{std::move(hooks)};

  service->HandleModelInferAsync(&ctx, &request, &reply, [&](grpc::Status) {
    callback_called.store(true);
  });

  EXPECT_FALSE(callback_called.load());
  expect_empty_infer_response(reply);

  std::shared_ptr<starpu_server::InferenceJob> job;
  EXPECT_FALSE(queue.try_pop(job));
}

TEST_F(InferenceServiceTest, ValidateInputsMismatchedRawContents)
{
  auto req = starpu_server::make_valid_request();
  req.add_raw_input_contents("extra");
  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(LongInputInferenceServiceTest, ValidateInputsDatatypeMismatch)
{
  auto req = starpu_server::make_valid_request();
  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(InferenceServiceTest, ValidateInputsShapeOverflow)
{
  starpu_server::InputSpec spec{
      {std::numeric_limits<int64_t>::max()},
      at::kFloat,
      starpu_server::to_raw_data<float>({1.0F})};
  auto req = starpu_server::make_model_infer_request({spec});
  std::vector<torch::Tensor> inputs;
  auto status = service->validate_and_convert_inputs(&req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(
    InferenceServiceTest, SubmitJobAndWaitReturnsUnavailableWhenQueueShutdown)
{
  queue.shutdown();
  ref_outputs = {torch::zeros({1}, torch::TensorOptions().dtype(at::kFloat))};
  std::vector<torch::Tensor> inputs = {
      torch::tensor({kF1}, torch::TensorOptions().dtype(at::kFloat))};
  std::vector<torch::Tensor> outputs = {
      torch::tensor({kF2}, torch::TensorOptions().dtype(at::kFloat))};

  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;
  starpu_server::detail::TimingInfo timing_info{};
  auto status =
      service->submit_job_and_wait(inputs, outputs, breakdown, timing_info);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
  EXPECT_TRUE(outputs.empty());
}

TEST(InferenceServiceImpl, SubmitJobAsyncReturnsUnavailableWhenQueueIsNull)
{
  std::vector<torch::Tensor> ref_outputs = {
      torch::zeros({1}, torch::TensorOptions().dtype(at::kFloat))};
  starpu_server::InferenceServiceImpl service(
      nullptr, &ref_outputs, {at::kFloat});

  std::vector<torch::Tensor> inputs = {
      torch::tensor({kF1}, torch::TensorOptions().dtype(at::kFloat))};

  auto status = service.submit_job_async(
      inputs,
      [](grpc::Status, std::vector<torch::Tensor>,
         starpu_server::InferenceServiceImpl::LatencyBreakdown,
         starpu_server::detail::TimingInfo,
         std::optional<starpu_server::InferenceServiceImpl::AsyncFailureInfo>) {
      });

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
  EXPECT_EQ(status.error_message(), "Inference queue unavailable");
}

TEST(
    InferenceServiceImpl,
    SubmitJobAsyncReturnsFailedPreconditionWhenReferenceOutputsAreNull)
{
  starpu_server::InferenceQueue queue;
  starpu_server::InferenceServiceImpl service(&queue, nullptr, {at::kFloat});

  std::vector<torch::Tensor> inputs = {
      torch::tensor({kF1}, torch::TensorOptions().dtype(at::kFloat))};

  auto status = service.submit_job_async(
      inputs,
      [](grpc::Status, std::vector<torch::Tensor>,
         starpu_server::InferenceServiceImpl::LatencyBreakdown,
         starpu_server::detail::TimingInfo,
         std::optional<starpu_server::InferenceServiceImpl::AsyncFailureInfo>) {
      });

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_EQ(status.error_message(), "Reference outputs are unavailable");
}

TEST_F(
    InferenceServiceTest, SubmitJobAsyncLogsTraceEventWhenBatchingTracerEnabled)
{
  auto& tracer = starpu_server::BatchingTraceLogger::instance();
  tracer.configure(false, "");

  const auto trace_path = make_temp_trace_path();
  tracer.configure(true, trace_path.string());
  TraceFileGuard guard{tracer, trace_path};

  ref_outputs = {torch::zeros({1}, torch::TensorOptions().dtype(at::kFloat))};
  std::vector<torch::Tensor> inputs = {
      torch::tensor({kF1, kF2}, torch::TensorOptions().dtype(at::kFloat))};

  auto status = service->submit_job_async(
      inputs,
      [](grpc::Status, std::vector<torch::Tensor>,
         starpu_server::InferenceServiceImpl::LatencyBreakdown,
         starpu_server::detail::TimingInfo,
         std::optional<starpu_server::InferenceServiceImpl::AsyncFailureInfo>) {
      });
  ASSERT_TRUE(status.ok());

  std::shared_ptr<starpu_server::InferenceJob> enqueued_job;
  ASSERT_TRUE(queue.try_pop(enqueued_job));

  tracer.configure(false, "");

  std::ifstream stream(trace_path);
  ASSERT_TRUE(stream.is_open());
  const std::string content(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("\"name\":\"request_enqueued\""), std::string::npos);
  EXPECT_NE(content.find("\"request_id\":0"), std::string::npos);
}

TEST_F(InferenceServiceTest, SubmitJobAsyncReportsFailureInfoWhenJobFailed)
{
  starpu_server::InferenceJob::FailureInfo failure_info{};
  failure_info.stage = "execution";
  failure_info.reason = "bad_input";
  failure_info.message = "invalid";
  failure_info.metrics_reported = true;

  auto worker = prepare_job(
      {}, {}, [failure_info](starpu_server::InferenceJob& job) mutable {
        job.set_failure_info(std::move(failure_info));
      });

  std::vector<torch::Tensor> inputs = {
      torch::tensor({kF1}, torch::TensorOptions().dtype(at::kFloat))};

  struct CallbackResult {
    grpc::Status status;
    std::optional<starpu_server::InferenceServiceImpl::AsyncFailureInfo>
        failure_info;
  };

  auto promise = std::make_shared<std::promise<CallbackResult>>();
  auto future = promise->get_future();

  auto submit_status = service->submit_job_async(
      inputs,
      [promise](
          grpc::Status status, std::vector<torch::Tensor>,
          starpu_server::InferenceServiceImpl::LatencyBreakdown,
          starpu_server::detail::TimingInfo,
          std::optional<starpu_server::InferenceServiceImpl::AsyncFailureInfo>
              failure_info) {
        promise->set_value(
            CallbackResult{std::move(status), std::move(failure_info)});
      });
  ASSERT_TRUE(submit_status.ok());

  CallbackResult result = future.get();
  worker.join();

  EXPECT_FALSE(result.status.ok());
  EXPECT_EQ(result.status.error_code(), grpc::StatusCode::INTERNAL);
  EXPECT_NE(result.status.error_message().find("bad_input"), std::string::npos);
  EXPECT_NE(result.status.error_message().find("invalid"), std::string::npos);
  ASSERT_TRUE(result.failure_info.has_value());
  EXPECT_EQ(result.failure_info->stage, "execution");
  EXPECT_EQ(result.failure_info->reason, "bad_input");
  EXPECT_TRUE(result.failure_info->metrics_reported);
}

TEST_F(
    InferenceServiceTest, SubmitJobAsyncFormatsFailureMessageForMissingFields)
{
  struct Case {
    std::string stage;
    std::string reason;
    std::string message;
    std::string expected_error;
  };

  const std::vector<Case> cases = {
      {"execution", "bad_input", "", "Inference failed (bad_input)"},
      {"preprocess", "", "invalid", "Inference failed: invalid"},
      {"postprocess", "", "", "Inference failed"},
  };

  std::vector<torch::Tensor> inputs = {
      torch::tensor({kF1}, torch::TensorOptions().dtype(at::kFloat))};

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.expected_error);
    starpu_server::InferenceJob::FailureInfo failure_info{};
    failure_info.stage = test_case.stage;
    failure_info.reason = test_case.reason;
    failure_info.message = test_case.message;
    failure_info.metrics_reported = true;

    auto worker = prepare_job(
        {}, {}, [failure_info](starpu_server::InferenceJob& job) mutable {
          job.set_failure_info(std::move(failure_info));
        });

    struct CallbackResult {
      grpc::Status status;
      std::optional<starpu_server::InferenceServiceImpl::AsyncFailureInfo>
          failure_info;
    };

    auto promise = std::make_shared<std::promise<CallbackResult>>();
    auto future = promise->get_future();

    auto submit_status = service->submit_job_async(
        inputs,
        [promise](
            grpc::Status status, std::vector<torch::Tensor>,
            starpu_server::InferenceServiceImpl::LatencyBreakdown,
            starpu_server::detail::TimingInfo,
            std::optional<starpu_server::InferenceServiceImpl::AsyncFailureInfo>
                failure_info) {
          promise->set_value(
              CallbackResult{std::move(status), std::move(failure_info)});
        });
    ASSERT_TRUE(submit_status.ok());

    CallbackResult result = future.get();
    worker.join();

    EXPECT_FALSE(result.status.ok());
    EXPECT_EQ(result.status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(result.status.error_message(), test_case.expected_error);
    ASSERT_TRUE(result.failure_info.has_value());
    EXPECT_EQ(result.failure_info->stage, test_case.stage);
    EXPECT_EQ(result.failure_info->reason, test_case.reason);
    EXPECT_TRUE(result.failure_info->metrics_reported);
  }
}

TEST_F(
    InferenceServiceTest,
    HandleModelInferAsyncInvokesCallbackWhenSubmitJobAsyncFails)
{
  auto request = starpu_server::make_valid_request();
  queue.shutdown();

  bool callback_called = false;
  grpc::Status callback_status;

  service->HandleModelInferAsync(
      &ctx, &request, &reply, [&](grpc::Status status) {
        callback_called = true;
        callback_status = std::move(status);
      });

  EXPECT_TRUE(callback_called);
  EXPECT_FALSE(callback_status.ok());
  EXPECT_EQ(callback_status.error_code(), grpc::StatusCode::UNAVAILABLE);
  expect_empty_infer_response(reply);
}

TEST_F(
    InferenceServiceTest,
    HandleModelInferAsyncSkipsCallbackWhenCancelledAfterSubmitFailure)
{
  auto request = starpu_server::make_valid_request();
  queue.shutdown();
  std::atomic<bool> callback_called{false};

  starpu_server::InferenceServiceImpl::HandleModelInferAsyncTestHooks hooks;
  hooks.on_submit_job_async_done =
      [](const std::shared_ptr<std::atomic<bool>>& cancel_flag,
         const grpc::Status& status) {
        if (!status.ok()) {
          cancel_flag->store(true, std::memory_order_release);
        }
      };
  HandleModelInferAsyncHooksGuard guard{std::move(hooks)};

  service->HandleModelInferAsync(&ctx, &request, &reply, [&](grpc::Status) {
    callback_called.store(true);
  });

  EXPECT_FALSE(callback_called.load());
  expect_empty_infer_response(reply);
}

TEST_F(
    InferenceServiceTest,
    HandleModelInferAsyncCancelsImmediatelyWithContextGuard)
{
  auto request = starpu_server::make_valid_request();
  std::atomic<int> callback_count{0};
  grpc::Status callback_status;
  std::function<void()> on_cancel;
  bool force_cancelled = true;

  starpu_server::InferenceServiceImpl::HandleModelInferAsyncTestHooks hooks;
  hooks.is_cancelled_override =
      [&force_cancelled](grpc::ServerContext*) -> std::optional<bool> {
    return force_cancelled;
  };
  hooks.on_cancel_ready =
      [&on_cancel](const std::function<void()>& cancel_handler) {
        on_cancel = cancel_handler;
      };
  HandleModelInferAsyncHooksGuard guard{std::move(hooks)};

  auto call_guard = std::make_shared<int>(1);

  service->HandleModelInferAsync(
      &ctx, &request, &reply,
      [&](grpc::Status status) {
        callback_status = std::move(status);
        callback_count.fetch_add(1);
      },
      call_guard);

  EXPECT_EQ(callback_count.load(), 1);
  EXPECT_EQ(callback_status.error_code(), grpc::StatusCode::CANCELLED);
  expect_empty_infer_response(reply);

  ASSERT_TRUE(static_cast<bool>(on_cancel));
  on_cancel();
  EXPECT_EQ(callback_count.load(), 1);
}

TEST_F(
    InferenceServiceTest,
    HandleModelInferAsyncCancelHandlerNoOpWhenNotCancelled)
{
  auto request = starpu_server::make_valid_request();
  request.clear_raw_input_contents();
  std::atomic<int> callback_count{0};
  grpc::Status callback_status;
  std::function<void()> on_cancel;

  starpu_server::InferenceServiceImpl::HandleModelInferAsyncTestHooks hooks;
  hooks.is_cancelled_override =
      [](grpc::ServerContext*) -> std::optional<bool> { return false; };
  hooks.on_cancel_ready =
      [&on_cancel](const std::function<void()>& cancel_handler) {
        on_cancel = cancel_handler;
      };
  HandleModelInferAsyncHooksGuard guard{std::move(hooks)};

  auto call_guard = std::make_shared<int>(1);

  service->HandleModelInferAsync(
      &ctx, &request, &reply,
      [&](grpc::Status status) {
        callback_status = std::move(status);
        callback_count.fetch_add(1);
      },
      call_guard);

  EXPECT_EQ(callback_count.load(), 1);
  EXPECT_EQ(callback_status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  expect_empty_infer_response(reply);

  ASSERT_TRUE(static_cast<bool>(on_cancel));
  on_cancel();
  EXPECT_EQ(callback_count.load(), 1);
}

TEST_F(
    InferenceServiceTest, HandleModelInferAsyncSkipsCallbackWhenCancelFlagSet)
{
  auto request = starpu_server::make_valid_request();
  std::atomic<bool> callback_called{false};

  starpu_server::InferenceServiceImpl::HandleModelInferAsyncTestHooks hooks;
  hooks.on_cancel_flag_created =
      [](const std::shared_ptr<std::atomic<bool>>& cancel_flag) {
        cancel_flag->store(true, std::memory_order_release);
      };
  HandleModelInferAsyncHooksGuard guard{std::move(hooks)};

  service->HandleModelInferAsync(&ctx, &request, &reply, [&](grpc::Status) {
    callback_called.store(true);
  });

  EXPECT_FALSE(callback_called.load());
  expect_empty_infer_response(reply);

  std::shared_ptr<starpu_server::InferenceJob> job;
  EXPECT_FALSE(queue.try_pop(job));
}

TEST_F(
    MetricsInferenceServiceTest,
    HandleModelInferAsyncUpdatesMetricsAndLatencyBreakdown)
{
  auto request = starpu_server::make_valid_request();
  auto expected_outputs = std::vector<torch::Tensor>{
      torch::tensor({kF2}, torch::TensorOptions().dtype(at::kFloat))};
  auto worker_outputs = expected_outputs;
  auto worker = prepare_job(
      expected_outputs, worker_outputs, [](starpu_server::InferenceJob& job) {
        job.timing_info().enqueued_time =
            starpu_server::MonotonicClock::time_point{};
        job.timing_info().last_enqueued_time = job.timing_info().enqueued_time;
        job.timing_info().callback_end_time =
            starpu_server::MonotonicClock::now() - std::chrono::milliseconds(1);
      });

  auto metrics = starpu_server::get_metrics();
  ASSERT_NE(metrics, nullptr);
  EXPECT_DOUBLE_EQ(metrics->counters().requests_total->Value(), 0.0);

  std::promise<grpc::Status> status_promise;
  auto status_future = status_promise.get_future();
  service->HandleModelInferAsync(
      &ctx, &request, &reply, [&status_promise](grpc::Status status) {
        status_promise.set_value(std::move(status));
      });

  auto status = status_future.get();
  worker.join();

  EXPECT_TRUE(status.ok());
  EXPECT_DOUBLE_EQ(metrics->counters().requests_total->Value(), 1.0);
  EXPECT_DOUBLE_EQ(reply.server_preprocess_ms(), 0.0);
  EXPECT_GT(reply.server_postprocess_ms(), 0.0);
  EXPECT_GE(reply.server_overall_ms(), 0.0);

  auto families = metrics->registry()->Collect();
  auto histogram_it = std::find_if(
      families.begin(), families.end(),
      [](const auto& family) { return family.name == "inference_latency_ms"; });
  ASSERT_NE(histogram_it, families.end());
  ASSERT_FALSE(histogram_it->metric.empty());
  const auto& histogram = histogram_it->metric.front().histogram;
  EXPECT_EQ(histogram.sample_count, 1);
  EXPECT_GT(histogram.sample_sum, 0.0);
}

TEST_F(
    MetricsInferenceServiceTest,
    HandleModelInferAsyncCountsSingleTerminalStatusWhenCancelRacesCompletion)
{
  auto request = starpu_server::make_valid_request();
  auto expected_outputs = std::vector<torch::Tensor>{
      torch::tensor({kF2}, torch::TensorOptions().dtype(at::kFloat))};
  auto worker = prepare_job(expected_outputs, expected_outputs);

  auto metrics = starpu_server::get_metrics();
  ASSERT_NE(metrics, nullptr);

  std::mutex cancel_handler_mutex;
  std::function<void()> cancel_handler;
  std::atomic<bool> force_cancelled{false};

  starpu_server::InferenceServiceImpl::HandleModelInferAsyncTestHooks
      model_hooks;
  model_hooks.is_cancelled_override =
      [&force_cancelled](grpc::ServerContext*) -> std::optional<bool> {
    return force_cancelled.load(std::memory_order_acquire);
  };
  model_hooks.on_cancel_ready = [&cancel_handler_mutex, &cancel_handler](
                                    const std::function<void()>& cancel_ready) {
    std::scoped_lock lock(cancel_handler_mutex);
    cancel_handler = cancel_ready;
  };

  starpu_server::InferenceServiceImpl::HandleAsyncInferCompletionTestHooks
      completion_hooks;
  completion_hooks.before_final_cancel_check =
      [&force_cancelled, &cancel_handler_mutex,
       &cancel_handler](const std::shared_ptr<std::atomic<bool>>&) {
        force_cancelled.store(true, std::memory_order_release);
        std::function<void()> local_handler;
        {
          std::scoped_lock lock(cancel_handler_mutex);
          local_handler = cancel_handler;
        }
        if (local_handler) {
          local_handler();
        }
      };

  HandleModelInferAsyncHooksGuard model_guard{std::move(model_hooks)};
  HandleAsyncInferCompletionHooksGuard completion_guard{
      std::move(completion_hooks)};

  std::promise<grpc::Status> status_promise;
  auto status_future = status_promise.get_future();
  std::atomic<int> callback_count{0};
  std::atomic<bool> double_callback{false};
  auto call_guard = std::make_shared<int>(1);

  service->HandleModelInferAsync(
      &ctx, &request, &reply,
      [&](grpc::Status status) {
        const int previous = callback_count.fetch_add(1);
        if (previous == 0) {
          status_promise.set_value(std::move(status));
        } else {
          double_callback.store(true, std::memory_order_relaxed);
        }
      },
      call_guard);

  ASSERT_EQ(
      status_future.wait_for(std::chrono::seconds(5)),
      std::future_status::ready);
  const auto status = status_future.get();
  worker.join();

  EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
  EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 1);
  EXPECT_FALSE(double_callback.load(std::memory_order_relaxed));

  const auto metric_families = metrics->registry()->Collect();
  const auto cancelled_label =
      starpu_server::monitoring::detail::status_code_label_for_test(
          static_cast<int>(grpc::StatusCode::CANCELLED));
  const auto ok_label =
      starpu_server::monitoring::detail::status_code_label_for_test(
          static_cast<int>(grpc::StatusCode::OK));
  EXPECT_DOUBLE_EQ(
      sum_counter_values_for_label(
          metric_families, "requests_by_status_total", "code", cancelled_label),
      1.0);
  EXPECT_DOUBLE_EQ(
      sum_counter_values_for_label(
          metric_families, "requests_by_status_total", "code", ok_label),
      0.0);
}

TEST_F(
    InferenceServiceTest,
    HandleModelInferAsyncPropagatesPopulateResponseFailureOnce)
{
  auto request = starpu_server::make_valid_request();
  static float dummy = 0.0F;
  const size_t huge_elems =
      std::numeric_limits<size_t>::max() / sizeof(float) + 1;
  auto huge_tensor = torch::from_blob(
      &dummy, {static_cast<int64_t>(huge_elems)},
      torch::TensorOptions().dtype(at::kFloat));
  auto worker = prepare_job(
      {torch::zeros({1}, torch::TensorOptions().dtype(at::kFloat))},
      {huge_tensor});

  std::promise<grpc::Status> status_promise;
  auto status_future = status_promise.get_future();
  std::atomic<int> callback_count{0};
  std::atomic<bool> double_callback{false};

  service->HandleModelInferAsync(
      &ctx, &request, &reply, [&](grpc::Status status) {
        int previous = callback_count.fetch_add(1);
        if (previous == 0) {
          status_promise.set_value(std::move(status));
        } else {
          double_callback.store(true);
        }
      });

  grpc::Status status = status_future.get();
  worker.join();

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(callback_count.load(), 1);
  EXPECT_FALSE(double_callback.load());
}

TEST_F(
    InferenceServiceTest,
    HandleModelInferAsyncConvertsCompletionExceptionToInternalError)
{
  auto request = starpu_server::make_valid_request();
  auto expected_outputs = std::vector<torch::Tensor>{
      torch::tensor({kF2}, torch::TensorOptions().dtype(at::kFloat))};
  auto worker = prepare_job(expected_outputs, expected_outputs);

  starpu_server::InferenceServiceImpl::HandleAsyncInferCompletionTestHooks
      hooks;
  hooks.before_final_cancel_check =
      [](const std::shared_ptr<std::atomic<bool>>&) {
        throw std::runtime_error("forced async completion failure");
      };
  HandleAsyncInferCompletionHooksGuard guard{std::move(hooks)};

  std::promise<grpc::Status> status_promise;
  auto status_future = status_promise.get_future();

  service->HandleModelInferAsync(
      &ctx, &request, &reply, [&status_promise](grpc::Status status) {
        status_promise.set_value(std::move(status));
      });

  const grpc::Status status = status_future.get();
  worker.join();

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
  EXPECT_EQ(status.error_message(), "Internal server error");
}

TEST_F(InferenceServiceTest, HandleModelInferAsyncIgnoresRepeatedCompletion)
{
  auto request = starpu_server::make_valid_request();
  auto expected_outputs = std::vector<torch::Tensor>{
      torch::tensor({kF2}, torch::TensorOptions().dtype(at::kFloat))};
  ref_outputs = expected_outputs;

  std::atomic<int> callback_count{0};
  std::atomic<bool> double_callback{false};
  std::promise<grpc::Status> status_promise;
  auto status_future = status_promise.get_future();

  std::jthread worker([&]() {
    std::shared_ptr<starpu_server::InferenceJob> job;
    if (!queue.wait_and_pop(job)) {
      return;
    }
    auto worker_outputs = expected_outputs;
    job->get_on_complete()(worker_outputs, 0.0);
    job->get_on_complete()(worker_outputs, 0.0);
  });

  service->HandleModelInferAsync(
      &ctx, &request, &reply, [&](grpc::Status status) {
        int previous = callback_count.fetch_add(1);
        if (previous == 0) {
          status_promise.set_value(std::move(status));
        } else {
          double_callback.store(true);
        }
      });

  grpc::Status status = status_future.get();
  worker.join();

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(callback_count.load(), 1);
  EXPECT_FALSE(double_callback.load());
  ASSERT_EQ(reply.outputs_size(), 1);
  const auto& out_tensor = reply.outputs(0);
  const auto expected_sizes = expected_outputs[0].sizes();
  ASSERT_EQ(out_tensor.shape_size(), expected_sizes.size());
  for (int i = 0; i < out_tensor.shape_size(); ++i) {
    EXPECT_EQ(out_tensor.shape(i), expected_sizes[static_cast<size_t>(i)]);
  }

  ASSERT_EQ(reply.raw_output_contents_size(), 1);
  auto expected_flat = expected_outputs[0].contiguous();
  ASSERT_EQ(expected_flat.scalar_type(), at::kFloat);
  const auto* expected_ptr = expected_flat.data_ptr<float>();
  const size_t expected_bytes =
      static_cast<size_t>(expected_flat.numel()) * expected_flat.element_size();
  EXPECT_EQ(reply.raw_output_contents(0).size(), expected_bytes);
  EXPECT_EQ(
      reply.raw_output_contents(0),
      std::string(reinterpret_cast<const char*>(expected_ptr), expected_bytes));
}

TEST_F(
    InferenceServiceTest,
    HandleModelInferAsyncCancelsDuringSubmitUnderConcurrentLoad)
{
  constexpr int kProducerThreads = 4;
  constexpr int kRequestsPerThread = 16;
  constexpr int kExpectedJobs = kProducerThreads * kRequestsPerThread;

  std::atomic<int> callback_count{0};

  starpu_server::InferenceServiceImpl::HandleModelInferAsyncTestHooks hooks;
  hooks.on_submit_job_async_done =
      [](const std::shared_ptr<std::atomic<bool>>& cancel_flag,
         const grpc::Status& status) {
        if (status.ok()) {
          cancel_flag->store(true, std::memory_order_release);
        }
      };
  HandleModelInferAsyncHooksGuard guard{std::move(hooks)};

  std::vector<std::jthread> producers;
  producers.reserve(kProducerThreads);
  for (int thread_index = 0; thread_index < kProducerThreads; ++thread_index) {
    producers.emplace_back([&]() {
      for (int request_index = 0; request_index < kRequestsPerThread;
           ++request_index) {
        auto request = starpu_server::make_valid_request();
        grpc::ServerContext context;
        inference::ModelInferResponse local_reply;
        service->HandleModelInferAsync(
            &context, &request, &local_reply, [&](grpc::Status) {
              callback_count.fetch_add(1, std::memory_order_relaxed);
            });
      }
    });
  }

  // Ensure all submit hooks had a chance to flip cancel flags before
  // completions are dispatched from queued jobs; otherwise this test is racy.
  producers.clear();

  int handled_jobs = 0;
  while (handled_jobs < kExpectedJobs) {
    std::shared_ptr<starpu_server::InferenceJob> job;
    ASSERT_TRUE(queue.wait_for_and_pop(job, std::chrono::milliseconds(500)));
    ASSERT_NE(job, nullptr);
    job->get_on_complete()({}, 0.0);
    ++handled_jobs;
  }

  EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 0);
}

TEST_F(
    InferenceServiceTest,
    HandleModelInferAsyncCancelsDuringCallbackUnderConcurrentLoad)
{
  constexpr int kProducerThreads = 4;
  constexpr int kRequestsPerThread = 16;
  constexpr int kExpectedJobs = kProducerThreads * kRequestsPerThread;

  ref_outputs = {torch::zeros({1}, torch::TensorOptions().dtype(at::kFloat))};
  const std::vector<torch::Tensor> worker_outputs = {
      torch::tensor({kF2}, torch::TensorOptions().dtype(at::kFloat))};

  std::atomic<int> callback_count{0};

  starpu_server::InferenceServiceImpl::HandleAsyncInferCompletionTestHooks
      hooks;
  hooks.before_final_cancel_check =
      [](const std::shared_ptr<std::atomic<bool>>& cancel_flag) {
        cancel_flag->store(true, std::memory_order_release);
      };
  HandleAsyncInferCompletionHooksGuard guard{std::move(hooks)};

  std::vector<std::jthread> producers;
  producers.reserve(kProducerThreads);
  for (int thread_index = 0; thread_index < kProducerThreads; ++thread_index) {
    producers.emplace_back([&]() {
      for (int request_index = 0; request_index < kRequestsPerThread;
           ++request_index) {
        auto request = starpu_server::make_valid_request();
        grpc::ServerContext context;
        inference::ModelInferResponse local_reply;
        service->HandleModelInferAsync(
            &context, &request, &local_reply, [&](grpc::Status) {
              callback_count.fetch_add(1, std::memory_order_relaxed);
            });
      }
    });
  }

  int handled_jobs = 0;
  while (handled_jobs < kExpectedJobs) {
    std::shared_ptr<starpu_server::InferenceJob> job;
    ASSERT_TRUE(queue.wait_for_and_pop(job, std::chrono::milliseconds(500)));
    ASSERT_NE(job, nullptr);
    auto outputs_copy = worker_outputs;
    job->get_on_complete()(outputs_copy, 0.0);
    ++handled_jobs;
  }

  producers.clear();
  EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 0);
}
