#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <span>
#include <string>

#include "monitoring/metrics.hpp"
#include "test_inference_service.hpp"

namespace {
constexpr float kF1 = 1.0F;
constexpr float kF2 = 2.0F;
constexpr float kF3 = 3.0F;
constexpr float kF4 = 4.0F;
constexpr int64_t kI10 = 10;
constexpr int64_t kI20 = 20;
constexpr int64_t kI30 = 30;
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

TEST(InferenceService, ValidateInputsSuccess)
{
  auto req = starpu_server::make_valid_request();
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1U);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(inputs[0].scalar_type(), at::kFloat);
  EXPECT_FLOAT_EQ(inputs[0][0][0].item<float>(), kF1);
}

TEST(InferenceService, ValidateInputsZeroCopyUsesRequestBuffer)
{
  constexpr size_t kElements = 1U << 10;
  std::vector<float> data(kElements, kF1);
  auto req = starpu_server::make_model_infer_request({
      {{static_cast<int64_t>(kElements)},
       at::kFloat,
       starpu_server::to_raw_data(data)},
  });

  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);

  std::vector<torch::Tensor> inputs;
  std::vector<std::shared_ptr<const void>> keep_alive;
  auto status = service.validate_and_convert_inputs(&req, inputs, &keep_alive);

  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1U);
  ASSERT_EQ(keep_alive.size(), 1U);
  EXPECT_EQ(
      inputs[0].data_ptr(), const_cast<void*>(static_cast<const void*>(
                                req.raw_input_contents(0).data())));
  EXPECT_EQ(
      inputs[0].nbytes(),
      static_cast<int64_t>(req.raw_input_contents(0).size()));
}

TEST(InferenceService, ValidateInputsKeepAliveSharesAlias)
{
  std::vector<float> data = {kF1, kF2, kF3, kF4};
  auto req = starpu_server::make_model_infer_request({
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data)},
  });

  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);

  std::vector<torch::Tensor> inputs;
  std::vector<std::shared_ptr<const void>> keep_alive;
  auto status = service.validate_and_convert_inputs(&req, inputs, &keep_alive);

  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1U);
  ASSERT_EQ(keep_alive.size(), 1U);
  EXPECT_EQ(inputs[0].data_ptr(), const_cast<void*>(keep_alive[0].get()));
  EXPECT_EQ(
      keep_alive[0].get(),
      static_cast<const void*>(req.raw_input_contents(0).data()));
}

TEST(InferenceService, ValidateInputsNonContiguous)
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
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1U);
  EXPECT_TRUE(inputs[0].is_contiguous());
  EXPECT_TRUE(torch::allclose(inputs[0], contig));
}

TEST(InferenceService, ValidateInputsMultipleDtypes)
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
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 2U);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(inputs[0].scalar_type(), at::kFloat);
  EXPECT_FLOAT_EQ(inputs[0][0][0].item<float>(), kF1);
  EXPECT_EQ(inputs[1].sizes(), (torch::IntArrayRef{3}));
  EXPECT_EQ(inputs[1].scalar_type(), at::kLong);
  EXPECT_EQ(inputs[1][0].item<int64_t>(), kI10);
}

TEST(InferenceService, ValidateInputsConfiguredShapeNoBatching)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  std::vector<std::vector<int64_t>> expected_dims = {{2, 2}};
  constexpr int kNoBatching = 0;
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types, expected_dims, kNoBatching);

  auto valid_req = starpu_server::make_shape_request({2, 2});
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&valid_req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1U);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));

  auto mismatched_dims_req = starpu_server::make_shape_request({2, 3});
  inputs.clear();
  status = service.validate_and_convert_inputs(&mismatched_dims_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  auto unexpected_rank_req = starpu_server::make_shape_request({2, 2, 2});
  inputs.clear();
  status = service.validate_and_convert_inputs(&unexpected_rank_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, ValidateInputsConfiguredShapeWithBatching)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  std::vector<std::vector<int64_t>> expected_dims = {{2, 2}};
  constexpr int kMaxBatchSize = 4;
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types, expected_dims, kMaxBatchSize);

  auto same_rank_req = starpu_server::make_shape_request({2, 2});
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&same_rank_req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1U);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));

  auto batch_only_req = starpu_server::make_shape_request({2});
  inputs.clear();
  status = service.validate_and_convert_inputs(&batch_only_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  auto explicit_batch_req = starpu_server::make_shape_request({3, 2, 2});
  inputs.clear();
  status = service.validate_and_convert_inputs(&explicit_batch_req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1U);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{3, 2, 2}));

  auto exceeding_batch_req = starpu_server::make_shape_request({5, 2, 2});
  inputs.clear();
  status = service.validate_and_convert_inputs(&exceeding_batch_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  auto mismatched_dims_req = starpu_server::make_shape_request({2, 3});
  inputs.clear();
  status = service.validate_and_convert_inputs(&mismatched_dims_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  auto mismatched_tail_req = starpu_server::make_shape_request({3, 2, 3});
  inputs.clear();
  status = service.validate_and_convert_inputs(&mismatched_tail_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, ValidateInputsConfiguredShapeRejectsTailLengthMismatch)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  std::vector<std::vector<int64_t>> expected_dims = {{2, 2}};
  constexpr int kMaxBatchSize = 4;
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types, expected_dims, kMaxBatchSize);

  auto mismatched_tail_length_req =
      starpu_server::make_shape_request({3, 2, 2, 2});
  std::vector<torch::Tensor> inputs;
  auto status =
      service.validate_and_convert_inputs(&mismatched_tail_length_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, ValidateInputsConfiguredShapeRejectsZeroRankBatch)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  std::vector<std::vector<int64_t>> expected_dims = {{}};
  constexpr int kMaxBatchSize = 4;
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types, expected_dims, kMaxBatchSize);

  auto zero_rank_req = starpu_server::make_shape_request({});
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&zero_rank_req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, ValidateInputsMismatchedCount)
{
  std::vector<float> data0 = {kF1, kF2, kF3, kF4};
  std::vector<int64_t> data1 = {kI10, kI20, kI30};
  auto req = starpu_server::make_model_infer_request({
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data0)},
      {{3}, at::kLong, starpu_server::to_raw_data(data1)},
  });
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
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
  if (!torch::cuda::is_available()) {
    GTEST_SKIP();
  }

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

TEST(InferenceService, ValidateInputsMismatchedRawContents)
{
  auto req = starpu_server::make_valid_request();
  req.add_raw_input_contents("extra");
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, ValidateInputsDatatypeMismatch)
{
  auto req = starpu_server::make_valid_request();
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kLong};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, ValidateInputsShapeOverflow)
{
  starpu_server::InputSpec spec{
      {std::numeric_limits<int64_t>::max()},
      at::kFloat,
      starpu_server::to_raw_data<float>({1.0F})};
  auto req = starpu_server::make_model_infer_request({spec});
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, SubmitJobAndWaitReturnsUnavailableWhenQueueShutdown)
{
  starpu_server::InferenceQueue queue;
  queue.shutdown();
  std::vector<torch::Tensor> ref_outputs = {
      torch::zeros({1}, torch::TensorOptions().dtype(at::kFloat))};
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs = {
      torch::tensor({kF1}, torch::TensorOptions().dtype(at::kFloat))};
  std::vector<torch::Tensor> outputs = {
      torch::tensor({kF2}, torch::TensorOptions().dtype(at::kFloat))};

  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;
  starpu_server::detail::TimingInfo timing_info{};
  auto status =
      service.submit_job_and_wait(inputs, outputs, breakdown, timing_info);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
  EXPECT_TRUE(outputs.empty());
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
            std::chrono::high_resolution_clock::time_point{};
        job.timing_info().callback_end_time =
            std::chrono::high_resolution_clock::now() -
            std::chrono::milliseconds(1);
      });

  auto metrics = starpu_server::get_metrics();
  ASSERT_NE(metrics, nullptr);
  EXPECT_DOUBLE_EQ(metrics->requests_total->Value(), 0.0);

  std::promise<grpc::Status> status_promise;
  auto status_future = status_promise.get_future();
  service->HandleModelInferAsync(
      &ctx, &request, &reply, [&status_promise](grpc::Status status) {
        status_promise.set_value(std::move(status));
      });

  auto status = status_future.get();
  worker.join();

  EXPECT_TRUE(status.ok());
  EXPECT_DOUBLE_EQ(metrics->requests_total->Value(), 1.0);
  EXPECT_DOUBLE_EQ(reply.server_preprocess_ms(), 0.0);
  EXPECT_GT(reply.server_postprocess_ms(), 0.0);
  EXPECT_GE(reply.server_overall_ms(), 0.0);

  auto families = metrics->registry->Collect();
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
