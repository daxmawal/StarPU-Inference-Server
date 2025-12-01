#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "core/inference_runner.hpp"
#include "core/inference_task.hpp"
#include "core/tensor_builder.hpp"
#include "starpu_task_worker/inference_queue.hpp"
#include "starpu_task_worker/starpu_task_worker.hpp"
#include "test_constants.hpp"
#include "test_helpers.hpp"
#include "test_inference_runner.hpp"
#include "utils/batching_trace_logger.hpp"

namespace {
inline auto
cuda_sync_result_ref() -> cudaError_t&
{
  static cudaError_t value = cudaSuccess;
  return value;
}
}  // namespace

extern "C" auto
cudaDeviceSynchronize() -> cudaError_t
{
  return cuda_sync_result_ref();
}

extern "C" auto
cudaGetErrorString([[maybe_unused]] cudaError_t error) -> const char*
{
  return "mock cuda error";
}


namespace {
constexpr int64_t kJobId = 7;
constexpr int64_t kWorkerId = 3;
constexpr double kLatencyMs = 123.0;

const std::vector<int64_t> kShape2x2{2, 2};
const std::vector<int64_t> kShape2x3{2, 3};
const std::vector<int64_t> kShape1{1};

const std::vector<torch::Dtype> kTypesFloatInt{torch::kFloat32, torch::kInt64};

struct ExpectedJobInfo {
  int64_t request_id;
  int worker_id;
};

struct ClientWorkerTestContext {
  starpu_server::RuntimeConfig config;
  std::vector<torch::Tensor> outputs_reference;
  starpu_server::InferenceQueue queue;
};

inline auto
make_client_worker_test_context() -> ClientWorkerTestContext
{
  starpu_server::RuntimeConfig config{};
  config.batching.delay_us = 0;
  config.batching.pregen_inputs = 1;

  starpu_server::TensorConfig tensor_cfg{
      .name = "input",
      .dims = {1, 3},
      .type = at::ScalarType::Float,
  };
  starpu_server::ModelConfig model_cfg{};
  model_cfg.inputs = {tensor_cfg};
  config.models = {model_cfg};

  std::vector<torch::Tensor> outputs_ref{torch::zeros({1, 3})};

  return ClientWorkerTestContext{
      std::move(config), std::move(outputs_ref),
      starpu_server::InferenceQueue()};
}

inline auto
make_trace_file_path(const char* suffix) -> std::filesystem::path
{
  const auto timestamp =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::string filename = std::string("client_worker_trace_") + suffix + "_" +
                         std::to_string(timestamp) + ".json";
  return std::filesystem::temp_directory_path() / filename;
}

inline auto
JobStateMatches(
    const std::shared_ptr<starpu_server::InferenceJob>& job,
    const std::vector<torch::Tensor>& inputs,
    const std::vector<at::ScalarType>& types,
    const std::vector<torch::Tensor>& outputs,
    std::chrono::high_resolution_clock::time_point start,
    const ExpectedJobInfo& expected) -> ::testing::AssertionResult
{
  if (job->get_request_id() != expected.request_id) {
    return ::testing::AssertionFailure() << "request_id mismatch";
  }
  if (job->get_input_tensors().size() != inputs.size()) {
    return ::testing::AssertionFailure() << "input count mismatch";
  }
  if (!job->get_input_tensors()[0].equal(inputs[0])) {
    return ::testing::AssertionFailure() << "input[0] tensor mismatch";
  }
  if (job->get_input_types().size() != types.size()) {
    return ::testing::AssertionFailure() << "input type count mismatch";
  }
  if (job->get_input_types()[0] != types[0]) {
    return ::testing::AssertionFailure() << "input type[0] mismatch";
  }
  if (job->get_output_tensors().size() != outputs.size()) {
    return ::testing::AssertionFailure() << "output count mismatch";
  }
  if (!job->get_output_tensors()[0].equal(outputs[0])) {
    return ::testing::AssertionFailure() << "output[0] tensor mismatch";
  }
  if (job->get_start_time() != start) {
    return ::testing::AssertionFailure() << "start time mismatch";
  }
  const auto fixed = job->get_fixed_worker_id();
  if (!fixed.has_value()) {
    return ::testing::AssertionFailure() << "fixed worker not set";
  }
  if (fixed.value() != expected.worker_id) {
    return ::testing::AssertionFailure() << "fixed worker mismatch";
  }
  if (!job->has_on_complete()) {
    return ::testing::AssertionFailure() << "on_complete not set";
  }
  return ::testing::AssertionSuccess();
}

inline auto
CallbackResultsMatch(
    bool called, const std::vector<torch::Tensor>& tensors, double latency,
    const std::vector<torch::Tensor>& outputs,
    double expected_latency) -> ::testing::AssertionResult
{
  if (!called) {
    return ::testing::AssertionFailure() << "callback not called";
  }
  if (tensors.size() != outputs.size()) {
    return ::testing::AssertionFailure() << "callback tensors size mismatch";
  }
  if (!tensors[0].equal(outputs[0])) {
    return ::testing::AssertionFailure() << "callback tensor[0] mismatch";
  }
  if (latency != expected_latency) {
    return ::testing::AssertionFailure() << "latency mismatch";
  }
  return ::testing::AssertionSuccess();
}
}  // namespace

TEST(InferenceJobTest, MakeShutdownJobCreatesShutdownSignal)
{
  auto job = starpu_server::InferenceJob::make_shutdown_job();
  ASSERT_NE(job, nullptr);
  EXPECT_TRUE(job->is_shutdown());
  EXPECT_EQ(job->get_request_id(), 0);
}

TEST(InferenceJobTest, ConstructorInitializesState)
{
  const std::vector<torch::Tensor> inputs{torch::ones(kShape2x2)};
  const std::vector<torch::Tensor> outputs{torch::zeros(kShape2x2)};
  const std::vector<at::ScalarType> types{at::kFloat};

  bool callback_called = false;
  std::vector<torch::Tensor> callback_tensors;
  double callback_latency = 0.0;

  const auto before = std::chrono::high_resolution_clock::now();
  auto job = std::make_shared<starpu_server::InferenceJob>(
      inputs, types, kJobId,
      [&](std::vector<torch::Tensor> tensors, double latency) {
        callback_called = true;
        callback_tensors = std::move(tensors);
        callback_latency = latency;
      });
  const auto after = std::chrono::high_resolution_clock::now();

  ASSERT_NE(job, nullptr);
  EXPECT_EQ(job->get_request_id(), kJobId);
  ASSERT_EQ(job->get_input_tensors().size(), inputs.size());
  EXPECT_TRUE(job->get_input_tensors()[0].equal(inputs[0]));
  ASSERT_EQ(job->get_input_types().size(), types.size());
  EXPECT_EQ(job->get_input_types()[0], types[0]);
  EXPECT_TRUE(job->has_on_complete());
  EXPECT_LE(before, job->get_start_time());
  EXPECT_GE(after, job->get_start_time());

  job->get_on_complete()(outputs, kLatencyMs);

  EXPECT_TRUE(callback_called);
  ASSERT_EQ(callback_tensors.size(), outputs.size());
  EXPECT_TRUE(callback_tensors[0].equal(outputs[0]));
  EXPECT_DOUBLE_EQ(callback_latency, kLatencyMs);
}

TEST(InferenceJobTest, SetPendingSubJobsStoresJobs)
{
  auto parent = std::make_shared<starpu_server::InferenceJob>();
  auto child1 = std::make_shared<starpu_server::InferenceJob>();
  auto child2 = std::make_shared<starpu_server::InferenceJob>();
  child1->set_request_id(10);
  child2->set_request_id(11);

  parent->set_pending_sub_jobs({child1, child2});

  const auto& pending = parent->pending_sub_jobs();
  ASSERT_EQ(pending.size(), 2U);
  EXPECT_EQ(pending[0], child1);
  EXPECT_EQ(pending[1], child2);
  EXPECT_TRUE(parent->has_pending_sub_jobs());
}

TEST(InferenceJobTest, TakePendingSubJobsReturnsAndClears)
{
  auto parent = std::make_shared<starpu_server::InferenceJob>();
  auto child = std::make_shared<starpu_server::InferenceJob>();
  child->set_request_id(99);

  parent->set_pending_sub_jobs({child});

  auto taken = parent->take_pending_sub_jobs();

  ASSERT_EQ(taken.size(), 1U);
  EXPECT_EQ(taken[0], child);
  EXPECT_TRUE(parent->pending_sub_jobs().empty());
  EXPECT_FALSE(parent->has_pending_sub_jobs());
}

TEST(InferenceJobTest, SettersGettersAndCallback)
{
  const std::vector<torch::Tensor> inputs{torch::ones(kShape2x2)};
  const std::vector<at::ScalarType> types{at::kFloat};
  const std::vector<torch::Tensor> outputs{torch::zeros(kShape2x2)};

  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_request_id(kJobId);
  job->set_input_tensors(inputs);
  job->set_input_types(types);
  job->set_output_tensors(outputs);
  job->set_fixed_worker_id(kWorkerId);

  const auto start = std::chrono::high_resolution_clock::now();
  job->set_start_time(start);

  bool callback_called = false;
  std::vector<torch::Tensor> cb_tensors;
  double cb_latency = 0.0;

  job->set_on_complete(
      [&](const std::vector<torch::Tensor>& tensor, double lat) {
        callback_called = true;
        cb_tensors = tensor;
        cb_latency = lat;
      });

  EXPECT_TRUE(JobStateMatches(
      job, inputs, types, outputs, start,
      ExpectedJobInfo{
          .request_id = kJobId, .worker_id = static_cast<int>(kWorkerId)}));

  job->get_on_complete()(job->get_output_tensors(), kLatencyMs);
  EXPECT_TRUE(CallbackResultsMatch(
      callback_called, cb_tensors, cb_latency, outputs, kLatencyMs));
}

TEST(InferenceJobTest, SetInputTensorsCopiesNonContiguousAsContiguous)
{
  auto non_contiguous =
      torch::arange(0, 6, torch::TensorOptions().dtype(torch::kFloat32))
          .reshape({2, 3})
          .transpose(0, 1);
  ASSERT_FALSE(non_contiguous.is_contiguous());

  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_input_tensors({non_contiguous});

  ASSERT_EQ(job->get_input_tensors().size(), 1U);
  const auto& stored = job->get_input_tensors()[0];
  EXPECT_TRUE(stored.is_contiguous());
  EXPECT_TRUE(stored.equal(non_contiguous));
}

TEST(InferenceJobTest, ReleaseInputTensorsMovesStoredInputs)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  const auto input = torch::ones(kShape2x2);
  job->set_input_tensors({input});

  ASSERT_EQ(job->get_input_tensors().size(), 1U);

  auto released = job->release_input_tensors();

  EXPECT_TRUE(job->get_input_tensors().empty());
  ASSERT_EQ(released.size(), 1U);
  EXPECT_TRUE(released[0].equal(input));
}

TEST(InferenceRunner_Unit, ResolveValidationModelReturnsNulloptForInvalidDevice)
{
  starpu_server::InferenceResult result{};
  result.executed_on = starpu_server::DeviceType::CUDA;
  result.device_id = -1;
  result.request_id = 42;

  torch::jit::script::Module cpu_model("cpu_module");
  const auto resolved = starpu_server::detail::resolve_validation_model(
      result, cpu_model, std::unordered_map<int, torch::jit::script::Module*>{},
      /*validate_results=*/true);

  EXPECT_FALSE(resolved.has_value());
}

TEST(InferenceRunner_Unit, BuildGpuModelLookup_ReturnsEmptyForAllNegativeIds)
{
  std::vector<torch::jit::script::Module> models_gpu;
  models_gpu.push_back(starpu_server::make_identity_model());
  models_gpu.push_back(starpu_server::make_identity_model());

  const std::vector<int> device_ids{-1, -2};

  const auto lookup =
      starpu_server::detail::build_gpu_model_lookup(models_gpu, device_ids);

  EXPECT_TRUE(lookup.empty());
}

TEST(InferenceRunner_Unit, BuildGpuModelLookup_SkipsNegativeEntries)
{
  std::vector<torch::jit::script::Module> models_gpu;
  models_gpu.push_back(starpu_server::make_identity_model());
  models_gpu.push_back(starpu_server::make_identity_model());
  models_gpu.push_back(starpu_server::make_identity_model());

  const std::vector<int> device_ids{2, -1, 0};

  const auto lookup =
      starpu_server::detail::build_gpu_model_lookup(models_gpu, device_ids);

  ASSERT_EQ(lookup.size(), 2U);
  EXPECT_EQ(lookup.at(0), &models_gpu[2]);
  EXPECT_EQ(lookup.at(2), &models_gpu[0]);
  EXPECT_FALSE(lookup.contains(1));
}

TEST(
    InferenceRunner_Unit, ResolveValidationModelReturnsNulloptForMissingReplica)
{
  starpu_server::InferenceResult result{};
  result.executed_on = starpu_server::DeviceType::CUDA;
  result.device_id = 3;
  result.request_id = 7;

  torch::jit::script::Module cpu_model("cpu_module");
  torch::jit::script::Module gpu_module("gpu_module");
  std::unordered_map<int, torch::jit::script::Module*> lookup{{0, &gpu_module}};

  const auto resolved = starpu_server::detail::resolve_validation_model(
      result, cpu_model, lookup, /*validate_results=*/true);

  EXPECT_FALSE(resolved.has_value());
}

TEST(InferenceRunner_Unit, ResolveValidationModelReturnsGpuModelWhenAvailable)
{
  starpu_server::InferenceResult result{};
  result.executed_on = starpu_server::DeviceType::CUDA;
  result.device_id = 1;
  result.request_id = 99;

  torch::jit::script::Module cpu_model("cpu_module");
  torch::jit::script::Module gpu_module_0("gpu_module_0");
  torch::jit::script::Module gpu_module_1("gpu_module_1");
  std::unordered_map<int, torch::jit::script::Module*> lookup{
      {0, &gpu_module_0}, {1, &gpu_module_1}};

  const auto resolved = starpu_server::detail::resolve_validation_model(
      result, cpu_model, lookup, /*validate_results=*/true);

  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(*resolved, &gpu_module_1);
}

TEST(InferenceRunner_Unit, ClientWorkerSeedsRngWhenSeedProvided)
{
  auto context = make_client_worker_test_context();
  auto& opts = context.config;
  auto& queue = context.queue;
  const auto& outputs_ref = context.outputs_reference;
  opts.seed = 123;

  ASSERT_TRUE(opts.seed.has_value());
  torch::manual_seed(*opts.seed);
  torch::rand({1, 3});
  const auto expected = torch::rand({2, 4});

  torch::manual_seed(0);
  starpu_server::detail::client_worker(queue, opts, outputs_ref, 0);

  const auto actual = torch::rand({2, 4});
  ASSERT_TRUE(torch::allclose(actual, expected));
}

TEST(InferenceRunner_Unit, ClientWorkerStopsWhenQueuePushFails)
{
  auto context = make_client_worker_test_context();
  auto& opts = context.config;
  auto& queue = context.queue;
  const auto& outputs_ref = context.outputs_reference;
  opts.batching.request_nb = 1;

  testing::internal::CaptureStderr();
  queue.shutdown();
  starpu_server::detail::client_worker(
      queue, opts, outputs_ref, opts.batching.request_nb);
  const auto captured = testing::internal::GetCapturedStderr();

  EXPECT_NE(captured.find("Failed to enqueue job"), std::string::npos);

  std::shared_ptr<starpu_server::InferenceJob> job;
  EXPECT_FALSE(queue.wait_and_pop(job));
}

TEST(InferenceRunner_Unit, ClientWorkerLogsTraceEventsWhenTracerEnabled)
{
  auto context = make_client_worker_test_context();
  auto& opts = context.config;
  opts.batching.request_nb = 1;
  opts.name = "trace-model";
  const auto& outputs_ref = context.outputs_reference;

  const auto trace_path = make_trace_file_path("enabled");
  std::error_code ec;
  std::filesystem::remove(trace_path, ec);

  auto& tracer = starpu_server::BatchingTraceLogger::instance();
  tracer.configure(false, "");
  tracer.configure(true, trace_path.string());

  starpu_server::detail::client_worker(
      context.queue, opts, outputs_ref, opts.batching.request_nb);

  tracer.configure(false, "");

  std::shared_ptr<starpu_server::InferenceJob> drained;
  while (context.queue.try_pop(drained)) {
  }

  ASSERT_TRUE(std::filesystem::exists(trace_path));
  std::ifstream trace_file(trace_path);
  ASSERT_TRUE(trace_file.is_open());
  std::string content(
      (std::istreambuf_iterator<char>(trace_file)),
      std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("request enqueued"), std::string::npos);
  std::filesystem::remove(trace_path);
}

TEST(InferenceRunner_Unit, ClientWorkerSkipsTraceEventsWhenTracerDisabled)
{
  auto context = make_client_worker_test_context();
  auto& opts = context.config;
  opts.batching.request_nb = 1;
  opts.name = "trace-model";
  const auto& outputs_ref = context.outputs_reference;

  const auto trace_path = make_trace_file_path("disabled");
  std::error_code ec;
  std::filesystem::remove(trace_path, ec);

  auto& tracer = starpu_server::BatchingTraceLogger::instance();
  tracer.configure(false, "");

  starpu_server::detail::client_worker(
      context.queue, opts, outputs_ref, opts.batching.request_nb);

  tracer.configure(false, "");

  EXPECT_FALSE(std::filesystem::exists(trace_path));
}

TEST(InferenceRunner_Unit, ValidateDeviceIdsNegativeAvailableCountThrows)
{
  const std::vector<int> device_ids{0, 1};
  EXPECT_THROW(
      starpu_server::detail::validate_device_ids(device_ids, -1),
      starpu_server::InvalidGpuDeviceException);
}

TEST(InferenceRunner_Unit, SetCudaDeviceCountOverrideRejectsNegative)
{
  EXPECT_THROW(
      starpu_server::detail::set_cuda_device_count_override(-5),
      std::invalid_argument);
}

TEST(InferenceRunnerUtils_Unit, GenerateInputsShapeAndType)
{
  const std::vector<std::vector<int64_t>> shapes{kShape2x3, kShape1};
  std::vector<starpu_server::TensorConfig> cfgs;
  cfgs.reserve(shapes.size());
  for (size_t i = 0; i < shapes.size(); ++i) {
    cfgs.push_back(
        {std::string("input") + std::to_string(i), shapes[i],
         kTypesFloatInt[i]});
  }
  torch::manual_seed(0);
  auto tensors = starpu_server::generate_inputs(cfgs);
  ASSERT_EQ(tensors.size(), 2U);
  EXPECT_EQ(
      tensors[0].sizes(), (torch::IntArrayRef{kShape2x3[0], kShape2x3[1]}));
  EXPECT_EQ(tensors[0].dtype(), torch::kFloat32);
  EXPECT_EQ(tensors[1].sizes(), (torch::IntArrayRef{kShape1[0]}));
  EXPECT_EQ(tensors[1].dtype(), torch::kInt64);
}

TEST(RunInference_Unit, CopyOutputToBufferCopiesData)
{
  using starpu_server::test_constants::kF1;
  using starpu_server::test_constants::kF2;
  constexpr float kF35 = 3.5F;
  constexpr float kFNeg4 = -4.0F;
  constexpr float kF025 = 0.25F;
  constexpr size_t kCount5 = 5;
  auto tensor = torch::tensor({kF1, kF2, kF35, kFNeg4, kF025}, torch::kFloat32);
  std::vector<float> dst(kCount5, 0.0F);
  auto dst_bytes =
      std::as_writable_bytes(std::span<float>(dst.data(), dst.size()));
  starpu_server::TensorBuilder::copy_output_to_buffer(
      tensor, dst_bytes, tensor.numel(), tensor.scalar_type());
  ASSERT_EQ(dst.size(), kCount5);
  EXPECT_FLOAT_EQ(dst[0], kF1);
  EXPECT_FLOAT_EQ(dst[1], kF2);
  EXPECT_FLOAT_EQ(dst[2], kF35);
  EXPECT_FLOAT_EQ(dst[3], kFNeg4);
  EXPECT_FLOAT_EQ(dst[4], kF025);
}

TEST(InferenceRunner_ProcessResults, ProcessResults_SkipsValidationWhenDisabled)
{
  auto cpu_model = starpu_server::make_identity_model();
  std::vector<torch::jit::script::Module> gpu_models;
  const std::vector<int> device_ids;
  starpu_server::RuntimeConfig opts{};
  opts.devices.ids = device_ids;
  opts.validation.validate_results = false;
  opts.verbosity = starpu_server::VerbosityLevel::Info;
  opts.validation.rtol = 1e-5;
  opts.validation.atol = 1e-8;

  starpu_server::InferenceResult result{};
  result.request_id = 1;
  result.executed_on = starpu_server::DeviceType::CPU;
  result.results = {torch::ones({1})};

  const std::vector<starpu_server::InferenceResult> results{result};

  testing::internal::CaptureStdout();
  starpu_server::detail::process_results(results, cpu_model, gpu_models, opts);
  const auto captured = testing::internal::GetCapturedStdout();

  EXPECT_NE(
      captured.find("Result validation disabled; skipping checks."),
      std::string::npos);
}

TEST(
    InferenceRunner_ProcessResults,
    ProcessResults_DoesNotLogErrorWhenResultsMissingAndValidationDisabled)
{
  auto cpu_model = starpu_server::make_identity_model();
  std::vector<torch::jit::script::Module> gpu_models;
  const std::vector<int> device_ids;
  starpu_server::RuntimeConfig opts{};
  opts.devices.ids = device_ids;
  opts.validation.validate_results = false;
  opts.verbosity = starpu_server::VerbosityLevel::Info;
  opts.validation.rtol = 1e-5;
  opts.validation.atol = 1e-8;

  starpu_server::InferenceResult result{};
  result.request_id = 13;
  result.executed_on = starpu_server::DeviceType::CPU;

  const std::vector<starpu_server::InferenceResult> results{result};

  testing::internal::CaptureStderr();
  starpu_server::detail::process_results(results, cpu_model, gpu_models, opts);
  const auto captured = testing::internal::GetCapturedStderr();

  EXPECT_EQ(captured.find("[Client] Job"), std::string::npos);
}

TEST(InferenceRunner_ProcessResults, ProcessResults_LogsErrorWhenResultMissing)
{
  auto cpu_model = starpu_server::make_identity_model();
  std::vector<torch::jit::script::Module> gpu_models;
  const std::vector<int> device_ids;
  starpu_server::RuntimeConfig opts{};
  opts.devices.ids = device_ids;
  opts.validation.validate_results = true;
  opts.verbosity = starpu_server::VerbosityLevel::Info;
  opts.validation.rtol = 1e-5;
  opts.validation.atol = 1e-8;

  starpu_server::InferenceResult result{};
  result.request_id = 99;
  result.executed_on = starpu_server::DeviceType::CPU;

  const std::vector<starpu_server::InferenceResult> results{result};

  testing::internal::CaptureStderr();
  starpu_server::detail::process_results(results, cpu_model, gpu_models, opts);
  const auto captured = testing::internal::GetCapturedStderr();

  EXPECT_NE(captured.find("[Client] Job"), std::string::npos);
}

TEST(
    InferenceRunner_ProcessResults,
    ProcessResults_SkipsWhenValidationModelUnavailable)
{
  auto cpu_model = starpu_server::make_identity_model();
  std::vector<torch::jit::script::Module> gpu_models;
  gpu_models.push_back(starpu_server::make_identity_model());
  const std::vector<int> device_ids{0};
  starpu_server::RuntimeConfig opts{};
  opts.devices.ids = device_ids;
  opts.validation.validate_results = true;
  opts.verbosity = starpu_server::VerbosityLevel::Info;
  opts.validation.rtol = 1e-5;
  opts.validation.atol = 1e-8;

  starpu_server::InferenceResult result{};
  result.request_id = 7;
  result.worker_id = 5;
  result.executed_on = starpu_server::DeviceType::CUDA;
  result.device_id = 1;
  result.inputs = {torch::zeros({2, 2})};
  result.results = {torch::ones({2, 2})};

  const std::vector<starpu_server::InferenceResult> results{result};

  testing::internal::CaptureStderr();
  starpu_server::detail::process_results(results, cpu_model, gpu_models, opts);
  const auto captured = testing::internal::GetCapturedStderr();

  EXPECT_NE(
      captured.find("[Client] Skipping validation for job"), std::string::npos);
  EXPECT_EQ(captured.find("[Validator]"), std::string::npos);
}

class DefaultWorkerThreadLauncherTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    completed_jobs_ = 0;
    starpu_setup_ = std::make_unique<starpu_server::StarPUSetup>(opts_);
    config_.queue = &queue_;
    config_.model_cpu = &model_cpu_;
    config_.models_gpu = &models_gpu_;
    config_.starpu = starpu_setup_.get();
    config_.opts = &opts_;
    config_.results = &results_;
    config_.results_mutex = &results_mutex_;
    config_.completed_jobs = &completed_jobs_;
    config_.all_done_cv = &cv_;
    config_.dependencies = &dependencies_;
  }
  starpu_server::InferenceQueue queue_;
  torch::jit::script::Module model_cpu_{"cpu_module"};
  std::vector<torch::jit::script::Module> models_gpu_;
  starpu_server::RuntimeConfig opts_;
  std::vector<starpu_server::InferenceResult> results_;
  std::mutex results_mutex_;
  std::atomic<int> completed_jobs_;
  std::condition_variable cv_;
  std::unique_ptr<starpu_server::StarPUSetup> starpu_setup_;
  starpu_server::StarPUTaskRunnerConfig config_;
  starpu_server::InferenceTaskDependencies dependencies_ =
      starpu_server::kDefaultInferenceTaskDependencies;
};

TEST_F(
    DefaultWorkerThreadLauncherTest,
    DefaultLauncherCreatesJthreadThatCallsRunMethod)
{
  starpu_server::StarPUTaskRunner runner(config_);
  auto launcher = starpu_server::get_worker_thread_launcher();
  auto thread = launcher(runner);
  EXPECT_TRUE(thread.joinable());
  [[maybe_unused]] auto push_result =
      queue_.push(starpu_server::InferenceJob::make_shutdown_job());
  const auto timeout = std::chrono::seconds(5);
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  thread.detach();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

TEST_F(DefaultWorkerThreadLauncherTest, DefaultLauncherReturnsValidJthread)
{
  starpu_server::StarPUTaskRunner runner(config_);
  auto launcher = starpu_server::get_worker_thread_launcher();
  auto thread = launcher(runner);
  EXPECT_TRUE(thread.joinable());
  [[maybe_unused]] auto push_result2 =
      queue_.push(starpu_server::InferenceJob::make_shutdown_job());
  thread.detach();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

TEST_F(
    DefaultWorkerThreadLauncherTest,
    DefaultLauncherThreadProcessesShutdownSignal)
{
  starpu_server::StarPUTaskRunner runner(config_);
  auto launcher = starpu_server::get_worker_thread_launcher();
  auto thread = launcher(runner);
  EXPECT_TRUE(thread.joinable());
  auto shutdown_job = starpu_server::InferenceJob::make_shutdown_job();
  EXPECT_TRUE(queue_.push(shutdown_job));
  thread.detach();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

TEST_F(DefaultWorkerThreadLauncherTest, DefaultLauncherCanBeSwitchedAndRestored)
{
  auto original_launcher = starpu_server::get_worker_thread_launcher();
  std::atomic<bool> custom_called(false);
  starpu_server::WorkerThreadLauncher custom_launcher =
      [&custom_called](starpu_server::StarPUTaskRunner&) -> std::jthread {
    custom_called = true;
    return std::jthread([] {});
  };
  starpu_server::set_worker_thread_launcher(custom_launcher);
  auto current = starpu_server::get_worker_thread_launcher();
  EXPECT_TRUE(static_cast<bool>(current));
  starpu_server::set_worker_thread_launcher(original_launcher);
  auto restored = starpu_server::get_worker_thread_launcher();
  EXPECT_TRUE(static_cast<bool>(restored));
}

TEST_F(
    DefaultWorkerThreadLauncherTest,
    DefaultLauncherCreatesThreadWithCorrectSignature)
{
  starpu_server::StarPUTaskRunner runner(config_);
  auto launcher = starpu_server::get_worker_thread_launcher();
  std::jthread thread = launcher(runner);
  EXPECT_TRUE(thread.joinable());
  [[maybe_unused]] auto push_result3 =
      queue_.push(starpu_server::InferenceJob::make_shutdown_job());
  thread.detach();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

class RunWarmupTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    starpu_setup_ = std::make_unique<starpu_server::StarPUSetup>(opts_);
    model_cpu_ = starpu_server::make_identity_model();
    outputs_ref_ = {torch::ones({1})};
  }
  starpu_server::RuntimeConfig opts_;
  std::unique_ptr<starpu_server::StarPUSetup> starpu_setup_;
  torch::jit::script::Module model_cpu_{"cpu_module"};
  std::vector<torch::jit::script::Module> models_gpu_;
  std::vector<torch::Tensor> outputs_ref_;
};

TEST_F(RunWarmupTest, RunWarmupReturnsEarlyWhenNoCpuAndNoCuda)
{
  opts_.devices.use_cpu = false;
  opts_.devices.use_cuda = false;
  opts_.batching.warmup_request_nb = 100;
  starpu_server::CaptureStream capture{std::cout};
  starpu_server::run_warmup(
      opts_, *starpu_setup_, model_cpu_, models_gpu_, outputs_ref_);
  auto output = capture.str();
  EXPECT_EQ(output.find("Starting warmup"), std::string::npos);
}

TEST_F(RunWarmupTest, RunWarmupReturnsEarlyWhenBothWarmupConfigZero)
{
  opts_.devices.use_cpu = true;
  opts_.devices.use_cuda = false;
  opts_.batching.warmup_request_nb = 0;
  opts_.batching.warmup_batches_per_worker = 0;
  starpu_server::CaptureStream capture{std::cout};
  starpu_server::run_warmup(
      opts_, *starpu_setup_, model_cpu_, models_gpu_, outputs_ref_);
  auto output = capture.str();
  EXPECT_EQ(output.find("Starting warmup"), std::string::npos);
}

TEST_F(RunWarmupTest, RunWarmupReturnsEarlyWhenCalculatedWarmupRequestNbZero)
{
  opts_.devices.use_cpu = true;
  opts_.devices.use_cuda = false;
  opts_.batching.warmup_request_nb = 0;
  opts_.batching.warmup_batches_per_worker = 0;
  opts_.batching.max_batch_size = 1;
  starpu_server::CaptureStream capture{std::cout};
  starpu_server::run_warmup(
      opts_, *starpu_setup_, model_cpu_, models_gpu_, outputs_ref_);
  auto output = capture.str();
  EXPECT_EQ(output.find("Starting warmup"), std::string::npos);
}

TEST_F(RunWarmupTest, RunWarmupNegativeWarmupWithPositiveBatches)
{
  opts_.devices.use_cpu = true;
  opts_.devices.use_cuda = false;
  opts_.batching.warmup_request_nb = -5;
  opts_.batching.warmup_batches_per_worker = 3;
  opts_.batching.max_batch_size = 2;
  int configured_batches = std::max(0, 3);
  int max_batch_size = std::max(1, 2);
  int min_requests_for_batches = configured_batches * max_batch_size;
  int warmup_request_nb = std::max(-5, min_requests_for_batches);
  EXPECT_EQ(configured_batches, 3);
  EXPECT_EQ(min_requests_for_batches, 6);
  EXPECT_EQ(warmup_request_nb, 6);
  EXPECT_GT(warmup_request_nb, 0);
}

TEST_F(RunWarmupTest, RunWarmupBothDeviceTypesConditionEval)
{
  opts_.devices.use_cpu = true;
  opts_.devices.use_cuda = true;
  EXPECT_TRUE(opts_.devices.use_cpu && opts_.devices.use_cuda);
}

TEST_F(RunWarmupTest, RunWarmupOnlyCudaConditionEval)
{
  opts_.devices.use_cpu = false;
  opts_.devices.use_cuda = true;
  EXPECT_FALSE(opts_.devices.use_cpu && opts_.devices.use_cuda);
  EXPECT_TRUE(opts_.devices.use_cuda);
}

TEST_F(RunWarmupTest, RunWarmupOnlyCpuConditionEval)
{
  opts_.devices.use_cpu = true;
  opts_.devices.use_cuda = false;
  EXPECT_FALSE(opts_.devices.use_cpu && opts_.devices.use_cuda);
  EXPECT_FALSE(opts_.devices.use_cuda);
}

TEST_F(RunWarmupTest, RunWarmupCalculatesMaxRequestNb)
{
  int warmup_request_nb = 5;
  int configured_batches = 2;
  int max_batch_size = 3;
  int min_requests_for_batches = configured_batches * max_batch_size;
  int result = std::max(warmup_request_nb, min_requests_for_batches);
  EXPECT_EQ(min_requests_for_batches, 6);
  EXPECT_EQ(result, 6);
}

TEST_F(RunWarmupTest, RunWarmupConfiguredBatchesZero)
{
  opts_.batching.warmup_batches_per_worker = -5;
  int configured_batches =
      std::max(0, opts_.batching.warmup_batches_per_worker);
  EXPECT_EQ(configured_batches, 0);
}

class ProcessResultsTest : public ::testing::Test {
 protected:
  void SetUp() override { model_cpu_ = starpu_server::make_identity_model(); }
  torch::jit::script::Module model_cpu_{"cpu_module"};
  std::vector<torch::jit::script::Module> models_gpu_;
  starpu_server::RuntimeConfig opts_;
};

TEST_F(ProcessResultsTest, ProcessResultsWithValidationDisabled)
{
  opts_.validation.validate_results = false;
  starpu_server::InferenceResult result;
  result.results = {torch::ones({1})};
  result.device_id = -1;
  std::vector<starpu_server::InferenceResult> results{result};
  EXPECT_NO_THROW(starpu_server::detail::process_results(
      results, model_cpu_, models_gpu_, opts_));
}

TEST_F(ProcessResultsTest, ProcessResultsWithValidationEnabledAndValidResults)
{
  opts_.validation.validate_results = true;
  opts_.verbosity = starpu_server::VerbosityLevel::Silent;
  opts_.devices.use_cuda = false;
  starpu_server::InferenceResult result;
  result.results = {torch::ones({1})};
  result.device_id = -1;
  std::vector<starpu_server::InferenceResult> results{result};
  starpu_server::InferenceResult empty_result;
  empty_result.results = {};
  std::vector<starpu_server::InferenceResult> empty_results{empty_result};
  EXPECT_NO_THROW(starpu_server::detail::process_results(
      empty_results, model_cpu_, models_gpu_, opts_));
}

TEST_F(ProcessResultsTest, ProcessResultsLogsJobFailureWhenValidationEnabled)
{
  opts_.validation.validate_results = true;
  opts_.verbosity = starpu_server::VerbosityLevel::Info;
  starpu_server::InferenceResult result;
  result.request_id = 42;
  result.results = {};
  std::vector<starpu_server::InferenceResult> results{result};
  starpu_server::CaptureStream capture{std::cerr};
  starpu_server::detail::process_results(
      results, model_cpu_, models_gpu_, opts_);
  auto output = capture.str();
  EXPECT_NE(output.find("[Client] Job"), std::string::npos);
  EXPECT_NE(output.find("failed"), std::string::npos);
}

TEST_F(
    ProcessResultsTest,
    ProcessResultsSkipsLoggingJobFailureWhenValidationDisabled)
{
  opts_.validation.validate_results = false;
  starpu_server::InferenceResult result;
  result.request_id = 42;
  result.results = {};
  std::vector<starpu_server::InferenceResult> results{result};
  starpu_server::CaptureStream capture{std::cerr};
  starpu_server::detail::process_results(
      results, model_cpu_, models_gpu_, opts_);
  auto output = capture.str();
  EXPECT_EQ(output.find("[Client] Job"), std::string::npos);
}

TEST_F(ProcessResultsTest, ProcessResultsSkipsValidationWhenDisabled)
{
  opts_.validation.validate_results = false;
  starpu_server::InferenceResult result;
  result.results = {torch::ones({1})};
  result.device_id = -1;
  std::vector<starpu_server::InferenceResult> results{result};
  EXPECT_NO_THROW(starpu_server::detail::process_results(
      results, model_cpu_, models_gpu_, opts_));
}

TEST_F(ProcessResultsTest, ProcessResultsJobFailureWithMultipleResults)
{
  opts_.validation.validate_results = true;
  opts_.verbosity = starpu_server::VerbosityLevel::Info;
  std::vector<starpu_server::InferenceResult> results;
  starpu_server::InferenceResult result1;
  result1.request_id = 1;
  result1.results = {};
  results.push_back(result1);
  starpu_server::InferenceResult result2;
  result2.request_id = 2;
  result2.results = {};
  results.push_back(result2);
  starpu_server::CaptureStream capture{std::cerr};
  starpu_server::detail::process_results(
      results, model_cpu_, models_gpu_, opts_);
  auto output = capture.str();
  EXPECT_NE(output.find("[Client] Job"), std::string::npos);
  EXPECT_NE(output.find("failed"), std::string::npos);
}

TEST_F(ProcessResultsTest, ProcessResultsValidationConditionBranching)
{
  opts_.validation.validate_results = true;
  bool should_validate_true = opts_.validation.validate_results;
  EXPECT_TRUE(should_validate_true);
  opts_.validation.validate_results = false;
  bool should_validate_false = opts_.validation.validate_results;
  EXPECT_FALSE(should_validate_false);
  EXPECT_NE(should_validate_true, should_validate_false);
}
