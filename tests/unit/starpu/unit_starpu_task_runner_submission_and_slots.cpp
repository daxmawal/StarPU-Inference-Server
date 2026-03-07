#include "unit_starpu_task_runner_support.hpp"

TEST_F(StarPUTaskRunnerFixture, SubmitInferenceTaskHandlesStarpuSubmitFailures)
{
  auto model_config = make_model_config(
      "trace_model", {make_tensor_config("input0", {1}, at::kFloat)},
      {make_tensor_config("output0", {1}, at::kFloat)});
  reset_runner_with_model(model_config, /*pool_size=*/1);

  starpu_test::ScopedStarpuDataAcquireOverride acquire_override(
      &NoOpStarpuDataAcquire);
  starpu_test::ScopedStarpuDataReleaseOverride release_override(
      &NoOpStarpuDataRelease);
  submit_override_calls = 0;
  starpu_test::ScopedStarpuTaskSubmitOverride submit_override(
      &AlwaysFailStarpuSubmit);

  const auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat);
  auto job = make_job(703, {torch::ones({1}, tensor_opts)}, {at::kFloat});
  job->set_model_name("trace_model");
  job->set_output_tensors({torch::zeros({1}, tensor_opts)});

  EXPECT_THROW(
      runner_->submit_inference_task(job),
      starpu_server::StarPUTaskSubmissionException);
  EXPECT_EQ(submit_override_calls.load(), 1);

  auto maybe_input_slot = starpu_setup_->input_pool().try_acquire();
  ASSERT_TRUE(maybe_input_slot.has_value());
  EXPECT_EQ(*maybe_input_slot, 0);
  starpu_setup_->input_pool().release(*maybe_input_slot);

  auto maybe_output_slot = starpu_setup_->output_pool().try_acquire();
  ASSERT_TRUE(maybe_output_slot.has_value());
  EXPECT_EQ(*maybe_output_slot, 0);
  starpu_setup_->output_pool().release(*maybe_output_slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsReturnsBatchWhenPoolMissing)
{
  const auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat);
  auto job = make_job(710, {torch::ones({1}, tensor_opts)}, {at::kFloat});

  constexpr int64_t kExpectedBatch = 3;

  const auto batch = starpu_server::StarPUTaskRunnerTestAdapter::
      validate_batch_and_copy_inputs_custom(
          runner_.get(), job, kExpectedBatch, /*input_pool=*/nullptr,
          /*input_slot=*/-1);
  EXPECT_EQ(batch, kExpectedBatch);
}

TEST_F(
    StarPUTaskRunnerFixture, ValidateBatchAndCopyInputsHandlesJobsWithoutInputs)
{
  starpu_server::RuntimeConfig pool_config{};
  pool_config.model = make_model_config("empty_inputs", {}, {});
  pool_config.batching.pool_size = 1;
  pool_config.batching.max_batch_size = 1;

  starpu_server::InputSlotPool input_pool(pool_config, /*slots=*/1);
  const int slot = input_pool.acquire();

  auto job = make_job(712, {});
  const int64_t batch = starpu_server::StarPUTaskRunnerTestAdapter::
      validate_batch_and_copy_inputs_custom(
          runner_.get(), job, /*batch=*/1, &input_pool, slot);
  EXPECT_EQ(batch, 1);
  EXPECT_TRUE(input_pool.base_ptrs(slot).empty());
  EXPECT_TRUE(input_pool.host_buffer_infos(slot).empty());

  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsThrowsOnBaseBufferMismatch)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {1}, at::kFloat)}, {});
  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  const auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat);
  auto job = make_job(711, {torch::ones({1}, tensor_opts)}, {at::kFloat});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  const auto original_infos = input_pool.host_buffer_infos(slot);
  auto& mutable_infos =
      const_cast<std::vector<starpu_server::InputSlotPool::HostBufferInfo>&>(
          input_pool.host_buffer_infos(slot));
  if (mutable_infos.empty()) {
    input_pool.release(slot);
    FAIL() << "Input slot must expose buffer metadata";
  }
  mutable_infos.pop_back();

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::
          validate_batch_and_copy_inputs_custom(
              runner_.get(), job, /*batch=*/1, &input_pool, slot),
      starpu_server::InputPoolMismatchException);

  mutable_infos = original_infos;
  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsThrowsWhenBatchOutsidePoolCapacity)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {1}, at::kFloat)}, {});
  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  const auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat);
  auto job = make_job(712, {torch::ones({1}, tensor_opts)}, {at::kFloat});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::
          validate_batch_and_copy_inputs_custom(
              runner_.get(), job, /*batch=*/0, &input_pool, slot),
      starpu_server::InputPoolCapacityException);

  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture, ValidateBatchAndCopyInputsRejectsUndefinedTensor)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {2, 2}, at::kFloat)}, {});
  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  torch::Tensor undefined;
  ASSERT_FALSE(undefined.defined());
  auto job = make_job(713, {undefined}, {at::kFloat});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::
          validate_batch_and_copy_inputs_custom(
              runner_.get(), job, /*batch=*/1, &input_pool, slot),
      starpu_server::InvalidInputTensorException);

  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsThrowsWhenElementSizeZero)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {3}, at::kFloat)}, {});

  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  auto job = make_job(
      21, {torch::ones({3}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  const auto& handles = input_pool.handles(slot);
  ASSERT_EQ(handles.size(), 1U);
  const auto handle = handles[0];
  ASSERT_NE(handle, nullptr);

  const auto snapshots = snapshot_vector_interfaces(handle);
  ASSERT_FALSE(snapshots.empty());
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface != nullptr) {
      snapshot.iface->elemsize = 0;
    }
  }

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::
          validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot),
      starpu_server::StarPUDataAcquireException);

  restore_vector_interfaces(snapshots);
  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsThrowsWhenTensorBytesMisaligned)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {3}, at::kFloat)}, {});

  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  auto job = make_job(
      22, {torch::ones({3}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  const auto& handles = input_pool.handles(slot);
  ASSERT_EQ(handles.size(), 1U);
  const auto handle = handles[0];
  ASSERT_NE(handle, nullptr);

  const auto snapshots = snapshot_vector_interfaces(handle);
  ASSERT_FALSE(snapshots.empty());
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface != nullptr) {
      snapshot.iface->elemsize = snapshot.elemsize + 1;
    }
  }

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::
          validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot),
      starpu_server::InvalidInputTensorException);

  restore_vector_interfaces(snapshots);
  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsThrowsWhenSlotCapacityExceeded)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {3}, at::kFloat)}, {});

  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  auto job = make_job(
      23, {torch::ones({3}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  const auto& handles = input_pool.handles(slot);
  ASSERT_EQ(handles.size(), 1U);
  const auto handle = handles[0];
  ASSERT_NE(handle, nullptr);

  const auto snapshots = snapshot_vector_interfaces(handle);
  ASSERT_FALSE(snapshots.empty());
  const auto expected_bytes =
      static_cast<std::size_t>(job->get_input_tensors()[0].nbytes());
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface != nullptr) {
      snapshot.iface->allocsize = expected_bytes - 1;
    }
  }

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::
          validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot),
      starpu_server::InputPoolCapacityException);

  restore_vector_interfaces(snapshots);
  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsAbortsRemainingCopiesAfterFailure)
{
  opts_.devices.use_cuda = false;
  auto model_config = make_model_config(
      "multi_input",
      {make_tensor_config("input0", {3}, at::kFloat),
       make_tensor_config("input1", {3}, at::kFloat),
       make_tensor_config("input2", {3}, at::kFloat)},
      {});
  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  const auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat);
  auto valid_a =
      torch::tensor(std::vector<float>{1.0F, 2.0F, 3.0F}, tensor_opts);
  torch::Tensor undefined_tensor;
  ASSERT_FALSE(undefined_tensor.defined());
  auto valid_b =
      torch::tensor(std::vector<float>{4.0F, 5.0F, 6.0F}, tensor_opts);

  auto job = make_job(
      714, {valid_a, undefined_tensor, valid_b},
      {at::kFloat, at::kFloat, at::kFloat});
  const auto& job_inputs = job->get_input_tensors();

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  const auto& base_ptrs = input_pool.base_ptrs(slot);
  const auto& buffer_infos = input_pool.host_buffer_infos(slot);
  ASSERT_EQ(base_ptrs.size(), job_inputs.size());

  constexpr unsigned char kSentinel = 0x7F;
  for (size_t idx = 0; idx < base_ptrs.size(); ++idx) {
    std::memset(base_ptrs[idx], kSentinel, buffer_infos[idx].bytes);
  }

  const auto last_tensor_bytes =
      static_cast<std::size_t>(job_inputs.back().nbytes());
  ASSERT_GT(last_tensor_bytes, 0U);
  ASSERT_GE(buffer_infos.back().bytes, last_tensor_bytes);
  std::vector<std::byte> sentinel_pattern(
      last_tensor_bytes, static_cast<std::byte>(kSentinel));

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::
          validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot),
      starpu_server::InvalidInputTensorException);

  auto* last_destination = base_ptrs.back();
  ASSERT_NE(last_destination, nullptr);
  EXPECT_TRUE(std::equal(
      last_destination, last_destination + last_tensor_bytes,
      sentinel_pattern.begin(), sentinel_pattern.end()));

  input_pool.release(slot);
}

TEST(SlotHandleLeaseTest, ConstructorSkipsNullHandles)
{
  const auto valid_handle =
      reinterpret_cast<starpu_data_handle_t>(static_cast<uintptr_t>(0x1));
  std::array<starpu_data_handle_t, 2> handles{nullptr, valid_handle};
  std::vector<starpu_data_handle_t> acquired;
  std::vector<starpu_data_handle_t> released;

  SlotHandleLeaseAcquireContext acquire_ctx;
  acquire_ctx.calls = &acquired;
  SlotHandleLeaseReleaseContext release_ctx;
  release_ctx.calls = &released;

  ScopedSlotHandleLeaseAcquireContext acquire_scope(acquire_ctx);
  ScopedSlotHandleLeaseReleaseContext release_scope(release_ctx);

  alignas(std::max_align_t) std::array<std::byte, 256> storage{};
  test_api::slot_handle_lease_construct(
      storage.data(), std::span<const starpu_data_handle_t>(handles), STARPU_W);

  ASSERT_EQ(acquired.size(), 1U);
  EXPECT_EQ(acquired[0], valid_handle);

  test_api::slot_handle_lease_destroy(storage.data());
  ASSERT_EQ(released.size(), 1U);
  EXPECT_EQ(released[0], valid_handle);
}

TEST(SlotHandleLeaseTest, ConstructorPropagatesAcquireFailures)
{
  const auto handle_ok =
      reinterpret_cast<starpu_data_handle_t>(static_cast<uintptr_t>(0x10));
  const auto handle_fail =
      reinterpret_cast<starpu_data_handle_t>(static_cast<uintptr_t>(0x20));
  std::array handles{handle_ok, handle_fail};

  std::vector<starpu_data_handle_t> acquired;
  std::vector<starpu_data_handle_t> released;

  SlotHandleLeaseAcquireContext acquire_ctx;
  acquire_ctx.calls = &acquired;
  acquire_ctx.fail_handle = handle_fail;
  acquire_ctx.fail_code = -7;
  SlotHandleLeaseReleaseContext release_ctx;
  release_ctx.calls = &released;

  ScopedSlotHandleLeaseAcquireContext acquire_scope(acquire_ctx);
  ScopedSlotHandleLeaseReleaseContext release_scope(release_ctx);

  alignas(std::max_align_t) std::array<std::byte, 256> storage{};
  EXPECT_THROW(
      test_api::slot_handle_lease_construct(
          storage.data(), std::span<const starpu_data_handle_t>(handles),
          STARPU_W),
      starpu_server::StarPUDataAcquireException);

  ASSERT_EQ(acquired.size(), 2U);
  EXPECT_EQ(acquired[0], handle_ok);
  EXPECT_EQ(acquired[1], handle_fail);

  ASSERT_EQ(released.size(), 1U);
  EXPECT_EQ(released[0], handle_ok);
}

TEST(BuildRequestIdsForTraceTest, ReturnsEmptyWhenJobMissing)
{
  namespace internal = starpu_server::task_runner_internal;

  std::shared_ptr<starpu_server::InferenceJob> missing_job;
  const auto ids = internal::build_request_ids_for_trace(missing_job);
  EXPECT_TRUE(ids.empty());
}

TEST(BuildRequestIdsForTraceTest, ReturnsStoredRequestIdWhenSubJobExpired)
{
  namespace internal = starpu_server::task_runner_internal;

  auto aggregated = std::make_shared<starpu_server::InferenceJob>();
  std::vector<starpu_server::InferenceJob::AggregatedSubJob> sub_jobs;
  {
    auto expired = std::make_shared<starpu_server::InferenceJob>();
    starpu_server::InferenceJob::AggregatedSubJob entry{};
    entry.job = expired;
    entry.request_id = -7;
    sub_jobs.push_back(entry);
  }

  aggregated->set_aggregated_sub_jobs(std::move(sub_jobs));

  const auto ids = internal::build_request_ids_for_trace(aggregated);
  ASSERT_EQ(ids.size(), 1U);
  EXPECT_EQ(ids[0], -7);
}

TEST(SliceOutputsForSubJobTest, MakesSlicesContiguousWhenSourceIsNot)
{
  using starpu_server::task_runner_internal::slice_outputs_for_sub_job;
  using starpu_server::task_runner_internal::SubJobSliceOptions;

  auto tensor =
      torch::arange(0, 12, torch::TensorOptions().dtype(torch::kFloat32));
  tensor = tensor.view({3, 4});
  auto non_contiguous = tensor.transpose(0, 1);
  ASSERT_FALSE(non_contiguous.is_contiguous());

  std::vector<torch::Tensor> aggregated{non_contiguous};
  const SubJobSliceOptions options{/*offset=*/1, /*batch_size=*/2};

  auto result = slice_outputs_for_sub_job(aggregated, options);
  ASSERT_EQ(result.outputs.size(), 1U);
  const auto& slice = result.outputs[0];

  EXPECT_TRUE(slice.is_contiguous());
  ASSERT_EQ(slice.dim(), 2);
  EXPECT_EQ(slice.size(0), 2);
  EXPECT_EQ(slice.size(1), 3);

  auto expected = non_contiguous.narrow(0, 1, 2).contiguous();
  EXPECT_TRUE(torch::allclose(slice, expected));
  EXPECT_EQ(result.processed_length, expected.size(0));
}

TEST(BatchSizeFromInputsTest, ReturnsOneWhenInputsEmpty)
{
  const std::vector<torch::Tensor> empty_inputs;
  EXPECT_EQ(test_api::batch_size_from_inputs(empty_inputs), 1U);
}

TEST(BatchSizeFromInputsTest, ReturnsOneWhenFirstInputIsScalar)
{
  auto scalar = torch::tensor(1);
  ASSERT_EQ(scalar.dim(), 0);

  const std::vector<torch::Tensor> inputs{scalar};
  EXPECT_EQ(test_api::batch_size_from_inputs(inputs), 1U);
}

TEST(ResolveBatchSizeForJobTest, ReturnsOneWhenJobMissing)
{
  std::shared_ptr<starpu_server::InferenceJob> missing_job;
  EXPECT_EQ(test_api::resolve_batch_size_for_job(nullptr, missing_job), 1);
}

TEST(CudaCopyBatchTest, FinalizeDisablesAsyncCopyWhenStreamSyncFails)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA runtime unavailable for finalize test";
  }

  auto* batch = test_api::cuda_copy_batch_create(true);
  ASSERT_NE(batch, nullptr);

  int src = 1;
  int dst = 0;
  const bool enqueued = test_api::cuda_copy_batch_enqueue(
      batch, reinterpret_cast<std::byte*>(&dst),
      reinterpret_cast<const std::byte*>(&src), sizeof(src),
      /*allow_async=*/true);
  ASSERT_TRUE(enqueued);
  ASSERT_TRUE(test_api::cuda_copy_batch_pending(batch));

  const auto stream = test_api::cuda_copy_batch_stream(batch);
  ASSERT_NE(stream, nullptr);

  {
    ScopedCudaStreamSyncFailure guard(stream);
    test_api::cuda_copy_batch_finalize(batch);
  }

  EXPECT_FALSE(test_api::cuda_copy_batch_enabled(batch));
  EXPECT_FALSE(test_api::cuda_copy_batch_pending(batch));
  EXPECT_GE(cuda_stream_sync_failure_count(), 1);
  test_api::cuda_copy_batch_destroy(batch);
}

TEST(CudaCopyBatchTest, ConstructorHandlesStreamCreationFailure)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA runtime unavailable for constructor test";
  }

  {
    ScopedCudaStreamCreateFailure guard;
    auto* batch = test_api::cuda_copy_batch_create(true);
    ASSERT_NE(batch, nullptr);
    EXPECT_TRUE(cuda_stream_create_failure_count() >= 1);
    EXPECT_EQ(test_api::cuda_copy_batch_stream(batch), nullptr);
    EXPECT_FALSE(test_api::cuda_copy_batch_enabled(batch));
    EXPECT_FALSE(test_api::cuda_copy_batch_pending(batch));
    test_api::cuda_copy_batch_destroy(batch);
  }
}

TEST(CudaCopyBatchTest, EnqueueDisablesAsyncCopyWhenMemcpyFails)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA runtime unavailable for enqueue test";
  }

  auto* batch = test_api::cuda_copy_batch_create(true);
  ASSERT_NE(batch, nullptr);
  ASSERT_TRUE(test_api::cuda_copy_batch_enabled(batch));
  const auto stream = test_api::cuda_copy_batch_stream(batch);
  ASSERT_NE(stream, nullptr);

  int src = 42;
  int dst = 0;

  {
    ScopedCudaMemcpyAsyncFailure guard(stream);
    const bool ok = test_api::cuda_copy_batch_enqueue(
        batch, reinterpret_cast<std::byte*>(&dst),
        reinterpret_cast<const std::byte*>(&src), sizeof(src),
        /*allow_async=*/true);
    EXPECT_FALSE(ok);
  }

  EXPECT_FALSE(test_api::cuda_copy_batch_enabled(batch));
  EXPECT_FALSE(test_api::cuda_copy_batch_pending(batch));
  EXPECT_GE(cuda_memcpy_failure_count(), 1);
  EXPECT_EQ(dst, 0);

  test_api::cuda_copy_batch_destroy(batch);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsUpdatesVectorNumelWhenNeeded)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {3}, at::kFloat)}, {});

  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  auto job = make_job(
      24, {torch::ones({3}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  const auto& handles = input_pool.handles(slot);
  ASSERT_EQ(handles.size(), 1U);
  const auto handle = handles[0];
  ASSERT_NE(handle, nullptr);

  auto snapshots = snapshot_vector_interfaces(handle);
  ASSERT_FALSE(snapshots.empty());
  const auto tensor_bytes =
      static_cast<std::size_t>(job->get_input_tensors()[0].nbytes());
  const auto adjusted_elem_size = tensor_bytes / 2;
  ASSERT_GT(adjusted_elem_size, 0U);
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface != nullptr) {
      snapshot.iface->elemsize = adjusted_elem_size;
    }
  }

  const auto expected_numel = tensor_bytes / adjusted_elem_size;
  const auto batch = starpu_server::StarPUTaskRunnerTestAdapter::
      validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot);
  EXPECT_EQ(batch, 1);
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface != nullptr) {
      EXPECT_EQ(
          snapshot.iface->nx,
          static_cast<decltype(snapshot.iface->nx)>(expected_numel));
    }
  }

  restore_vector_interfaces(snapshots);
  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsCopiesPendingJobInputsSequentially)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {2}, at::kFloat)}, {});

  opts_.batching.max_batch_size = 2;
  reset_runner_with_model(model_config, /*pool_size=*/2);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  auto job = make_job(
      30,
      {torch::tensor(
          std::vector<float>{1.0F, 2.0F},
          torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto pending = make_job(
      31,
      {torch::tensor(
          std::vector<float>{3.0F, 4.0F},
          torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  job->set_pending_sub_jobs({pending});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  const auto& handles = input_pool.handles(slot);
  ASSERT_EQ(handles.size(), 1U);
  const auto handle = handles[0];
  ASSERT_NE(handle, nullptr);

  auto snapshots = snapshot_vector_interfaces(handle);
  ASSERT_FALSE(snapshots.empty());

  const int64_t batch = starpu_server::StarPUTaskRunnerTestAdapter::
      validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot);
  EXPECT_EQ(batch, 1);

  const std::vector<float> expected{1.0F, 2.0F, 3.0F, 4.0F};
  auto base_ptr = input_pool.base_ptrs(slot).at(0);
  ASSERT_NE(base_ptr, nullptr);
  std::vector<float> actual(expected.size());
  std::memcpy(actual.data(), base_ptr, expected.size() * sizeof(float));
  for (size_t idx = 0; idx < expected.size(); ++idx) {
    EXPECT_FLOAT_EQ(actual[idx], expected[idx]);
  }

  const auto total_numel = expected.size();
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface != nullptr) {
      EXPECT_EQ(
          snapshot.iface->nx,
          static_cast<decltype(snapshot.iface->nx)>(total_numel));
    }
  }

  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsThrowsWhenPendingJobInputCountMismatch)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {2}, at::kFloat)}, {});

  opts_.batching.max_batch_size = 2;
  reset_runner_with_model(model_config, /*pool_size=*/2);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  auto job = make_job(
      32,
      {torch::tensor(
          std::vector<float>{1.0F, 2.0F},
          torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto pending = make_job(33, {}, {});

  job->set_pending_sub_jobs({pending});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::
          validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot),
      starpu_server::InconsistentInputTensorCountException);

  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsThrowsWhenPendingTensorUndefined)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {2}, at::kFloat)}, {});

  opts_.batching.max_batch_size = 2;
  reset_runner_with_model(model_config, /*pool_size=*/2);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  auto job = make_job(
      34,
      {torch::tensor(
          std::vector<float>{1.0F, 2.0F},
          torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto pending = make_job(35, {}, {at::kFloat});
  torch::Tensor undefined;
  pending->set_input_tensors({undefined});
  job->set_pending_sub_jobs({pending});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::
          validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot),
      starpu_server::InvalidInputTensorException);

  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsThrowsWhenPendingTensorExceedsSlotCapacity)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {2}, at::kFloat)}, {});

  opts_.batching.max_batch_size = 2;
  reset_runner_with_model(model_config, /*pool_size=*/2);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  auto job = make_job(
      36,
      {torch::tensor(
          std::vector<float>{1.0F, 2.0F},
          torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto pending = make_job(
      37,
      {torch::tensor(
          std::vector<float>{3.0F, 4.0F},
          torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  job->set_pending_sub_jobs({pending});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  auto& buffer_infos =
      const_cast<std::vector<starpu_server::InputSlotPool::HostBufferInfo>&>(
          input_pool.host_buffer_infos(slot));
  ASSERT_FALSE(buffer_infos.empty());
  const auto original_bytes = buffer_infos[0].bytes;
  const auto tensor_bytes =
      static_cast<std::size_t>(job->get_input_tensors()[0].nbytes());
  buffer_infos[0].bytes = tensor_bytes + tensor_bytes / 2;

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::
          validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot),
      starpu_server::InputPoolCapacityException);

  buffer_infos[0].bytes = original_bytes;
  input_pool.release(slot);
}

TEST_F(StarPUTaskRunnerFixture, ValidateBatchAndCopyInputsSkipsNullPendingJobs)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {2}, at::kFloat)}, {});

  opts_.batching.max_batch_size = 3;
  reset_runner_with_model(model_config, /*pool_size=*/3);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  auto tensor_from = [](float a, float b) {
    return torch::tensor(
        std::vector<float>{a, b}, torch::TensorOptions().dtype(torch::kFloat));
  };

  auto job = make_job(38, {tensor_from(1.0F, 2.0F)}, {at::kFloat});
  auto pending_a = make_job(39, {tensor_from(3.0F, 4.0F)}, {at::kFloat});
  std::shared_ptr<starpu_server::InferenceJob> pending_null;
  auto pending_b = make_job(40, {tensor_from(5.0F, 6.0F)}, {at::kFloat});

  job->set_pending_sub_jobs({pending_a, pending_null, pending_b});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();

  const int64_t batch = starpu_server::StarPUTaskRunnerTestAdapter::
      validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot);
  EXPECT_EQ(batch, 1);

  auto base_ptr = input_pool.base_ptrs(slot).at(0);
  ASSERT_NE(base_ptr, nullptr);
  const std::vector<float> expected{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  std::vector<float> actual(expected.size());
  std::memcpy(actual.data(), base_ptr, expected.size() * sizeof(float));
  for (size_t idx = 0; idx < expected.size(); ++idx) {
    EXPECT_FLOAT_EQ(actual[idx], expected[idx]);
  }

  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsInfersBatchFromTensorRank)
{
  constexpr int64_t kBatchSize = 2;
  opts_.batching.max_batch_size = kBatchSize;

  auto model_config = make_model_config(
      "ranked_input", {make_tensor_config("input0", {3}, at::kFloat)}, {});
  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  const auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat);
  auto job =
      make_job(34, {torch::ones({kBatchSize, 3}, tensor_opts)}, {at::kFloat});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();

  const int64_t batch = starpu_server::StarPUTaskRunnerTestAdapter::
      validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot);
  EXPECT_EQ(batch, kBatchSize);

  auto base_ptr = input_pool.base_ptrs(slot).at(0);
  ASSERT_NE(base_ptr, nullptr);
  std::vector<float> actual(static_cast<std::size_t>(kBatchSize * 3));
  std::memcpy(actual.data(), base_ptr, actual.size() * sizeof(actual.front()));
  for (const auto value : actual) {
    EXPECT_FLOAT_EQ(value, 1.0F);
  }

  input_pool.release(slot);
}

TEST_F(StarPUTaskRunnerFixture, MergeInputTensorsConcatenatesBatchedJobs)
{
  auto job_a = make_job(
      40,
      {torch::tensor(
          {{1.0F}, {2.0F}}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto job_b = make_job(
      41,
      {torch::tensor({{3.0F}}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs{job_a, job_b};
  const auto merged =
      starpu_server::StarPUTaskRunnerTestAdapter::merge_input_tensors(
          jobs, /*total_samples=*/3);

  ASSERT_EQ(merged.size(), 1U);
  const auto expected = torch::tensor(
      {{1.0F}, {2.0F}, {3.0F}}, torch::TensorOptions().dtype(torch::kFloat));
  EXPECT_TRUE(torch::equal(merged[0], expected));
}

TEST_F(StarPUTaskRunnerFixture, MergeInputTensorsThrowsWhenTotalSamplesMismatch)
{
  auto job_a = make_job(
      42,
      {torch::tensor(
          {{1.0F}, {2.0F}}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto job_b = make_job(
      43,
      {torch::tensor({{3.0F}}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs{job_a, job_b};

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::merge_input_tensors(
          jobs, /*total_samples=*/5),
      starpu_server::InvalidInputTensorException);
}

TEST_F(StarPUTaskRunnerFixture, MergeInputTensorsReturnsEmptyWhenNoJobs)
{
  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs;

  const auto merged =
      starpu_server::StarPUTaskRunnerTestAdapter::merge_input_tensors(
          jobs, /*total_samples=*/0);

  EXPECT_TRUE(merged.empty());
}

TEST_F(
    StarPUTaskRunnerFixture, MergeInputTensorsReturnOriginalWhenSingleJobBatch)
{
  auto job = make_job(
      46,
      {torch::tensor(
          {{5.0F}, {7.0F}}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs{job};

  const auto merged =
      starpu_server::StarPUTaskRunnerTestAdapter::merge_input_tensors(
          jobs, /*total_samples=*/0);

  ASSERT_EQ(merged.size(), 1U);
  const auto& original = job->get_input_tensors().front();
  EXPECT_EQ(merged.front().data_ptr(), original.data_ptr());
  EXPECT_TRUE(torch::equal(merged.front(), original));
}

TEST_F(
    StarPUTaskRunnerFixture,
    MergeInputTensorsThrowsWhenJobLacksExpectedTensorIndex)
{
  auto job_a = make_job(
      47,
      {torch::tensor(
           {{1.0F}, {2.0F}}, torch::TensorOptions().dtype(torch::kFloat)),
       torch::tensor(
           {{3.0F}, {4.0F}}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat, at::kFloat});
  auto job_b = make_job(
      48,
      {torch::tensor(
          {{5.0F}, {6.0F}}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs{job_a, job_b};

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::merge_input_tensors(
          jobs, /*total_samples=*/0),
      starpu_server::InconsistentInputTensorCountException);
}

TEST(ValidateTensorAgainstPrototype, RejectsUndefinedTensorBeforeBatching)
{
  torch::Tensor undefined_tensor;
  auto prototype =
      torch::ones({1, 1}, torch::TensorOptions().dtype(torch::kFloat));

  EXPECT_THROW(
      test_api::validate_tensor_against_prototype(undefined_tensor, prototype),
      starpu_server::InvalidInputTensorException);
}

TEST(ValidateTensorAgainstPrototype, RejectsRankMismatchDuringBatching)
{
  auto tensor =
      torch::ones({1, 2, 3}, torch::TensorOptions().dtype(torch::kFloat));
  auto prototype =
      torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat));

  EXPECT_THROW(
      test_api::validate_tensor_against_prototype(tensor, prototype),
      starpu_server::InvalidInputTensorException);
}

TEST(ValidateTensorAgainstPrototype, RejectsNonPositiveRankTensors)
{
  auto tensor =
      torch::tensor(1.0F, torch::TensorOptions().dtype(torch::kFloat));
  auto prototype =
      torch::tensor(0.0F, torch::TensorOptions().dtype(torch::kFloat));

  EXPECT_THROW(
      test_api::validate_tensor_against_prototype(tensor, prototype),
      starpu_server::InvalidInputTensorException);
}

TEST(ValidateTensorAgainstPrototype, RejectsShapeMismatchBeyondBatchDimension)
{
  auto tensor =
      torch::ones({2, 3}, torch::TensorOptions().dtype(torch::kFloat));
  auto prototype =
      torch::ones({2, 4}, torch::TensorOptions().dtype(torch::kFloat));

  EXPECT_THROW(
      test_api::validate_tensor_against_prototype(tensor, prototype),
      starpu_server::InvalidInputTensorException);
}

TEST(ValidatePrototypeTensor, RejectsUndefinedTensorBeforeBatching)
{
  torch::Tensor undefined_tensor;
  EXPECT_THROW(
      test_api::validate_prototype_tensor(undefined_tensor),
      starpu_server::InvalidInputTensorException);
}

TEST(ValidatePrototypeTensor, RejectsNonPositiveRankTensors)
{
  auto scalar =
      torch::tensor(1.0F, torch::TensorOptions().dtype(torch::kFloat));
  EXPECT_THROW(
      test_api::validate_prototype_tensor(scalar),
      starpu_server::InvalidInputTensorException);
}

TEST_F(
    StarPUTaskRunnerFixture, MergeInputMemoryHoldersPreservesOriginalOrdering)
{
  auto job_a = make_job(44, {});
  auto owner_a0 = std::make_shared<int>(7);
  auto owner_a1 = std::make_shared<int>(9);
  job_a->set_input_memory_holders(
      {std::shared_ptr<const void>(owner_a0, owner_a0.get()),
       std::shared_ptr<const void>(owner_a1, owner_a1.get())});

  auto job_b = make_job(45, {});
  auto owner_b0 = std::make_shared<int>(11);
  job_b->set_input_memory_holders(
      {std::shared_ptr<const void>(owner_b0, owner_b0.get())});

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs{job_a, job_b};
  const auto holders =
      starpu_server::StarPUTaskRunnerTestAdapter::merge_input_memory_holders(
          jobs);

  ASSERT_EQ(holders.size(), 3U);
  EXPECT_EQ(holders[0].get(), owner_a0.get());
  EXPECT_EQ(holders[1].get(), owner_a1.get());
  EXPECT_EQ(holders[2].get(), owner_b0.get());
}

TEST_F(StarPUTaskRunnerFixture, EnqueuePreparedJobDeliversJobToWaiter)
{
  auto job = make_job(46, {});

  starpu_server::StarPUTaskRunnerTestAdapter::enqueue_prepared_job(
      runner_.get(), job);

  auto dequeued =
      starpu_server::StarPUTaskRunnerTestAdapter::wait_for_prepared_job(
          runner_.get());
  ASSERT_TRUE(dequeued);
  EXPECT_EQ(dequeued, job);
}

TEST_F(
    StarPUTaskRunnerFixture,
    EnqueuePreparedJobDoesNotIncrementInflightWhenLimitIsZero)
{
  ASSERT_EQ(opts_.batching.max_inflight_tasks, 0U);
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);

  auto job = make_job(461, {});
  starpu_server::StarPUTaskRunnerTestAdapter::enqueue_prepared_job(
      runner_.get(), job);

  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);
}

TEST_F(
    StarPUTaskRunnerFixture, EnqueuePreparedJobIncrementsInflightWhenLimitSet)
{
  opts_.batching.max_inflight_tasks = 10;
  runner_.reset();
  starpu_setup_.reset();
  starpu_setup_ = std::make_unique<starpu_server::StarPUSetup>(opts_);
  config_.starpu = starpu_setup_.get();
  config_.opts = &opts_;
  runner_ = std::make_unique<starpu_server::StarPUTaskRunner>(config_);

  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_max_inflight_tasks(
          runner_.get()),
      10U);
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);

  auto job = make_job(462, {});
  starpu_server::StarPUTaskRunnerTestAdapter::enqueue_prepared_job(
      runner_.get(), job);

  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    EnqueuePreparedJobIncrementsInflightForMultipleJobs)
{
  opts_.batching.max_inflight_tasks = 10;
  runner_.reset();
  starpu_setup_.reset();
  starpu_setup_ = std::make_unique<starpu_server::StarPUSetup>(opts_);
  config_.starpu = starpu_setup_.get();
  config_.opts = &opts_;
  runner_ = std::make_unique<starpu_server::StarPUTaskRunner>(config_);

  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);

  for (int i = 0; i < 5; ++i) {
    auto job = make_job(463 + i, {});
    starpu_server::StarPUTaskRunnerTestAdapter::enqueue_prepared_job(
        runner_.get(), job);
  }

  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      5U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    EnqueuePreparedJobDoesNotIncrementInflightForNullJob)
{
  opts_.batching.max_inflight_tasks = 10;
  runner_.reset();
  starpu_setup_.reset();
  starpu_setup_ = std::make_unique<starpu_server::StarPUSetup>(opts_);
  config_.starpu = starpu_setup_.get();
  config_.opts = &opts_;
  runner_ = std::make_unique<starpu_server::StarPUTaskRunner>(config_);

  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);

  starpu_server::StarPUTaskRunnerTestAdapter::enqueue_prepared_job(
      runner_.get(), nullptr);

  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ReleasePendingJobsClearsInputsForAdditionalJobsOnly)
{
  auto master_job = make_job(
      47, {torch::tensor({5.0F}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto master_holder = std::make_shared<int>(21);
  master_job->set_input_memory_holders(
      {std::shared_ptr<const void>(master_holder, master_holder.get())});

  auto pending_job = make_job(
      48, {torch::tensor({9.0F}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto pending_holder = std::make_shared<int>(22);
  pending_job->set_input_memory_holders(
      {std::shared_ptr<const void>(pending_holder, pending_holder.get())});

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> pending_jobs{
      pending_job};
  starpu_server::StarPUTaskRunnerTestAdapter::release_pending_jobs(
      master_job, pending_jobs);

  EXPECT_TRUE(pending_jobs.empty());
  EXPECT_TRUE(pending_job->get_input_tensors().empty());
  EXPECT_TRUE(pending_job->get_input_memory_holders().empty());
  EXPECT_FALSE(master_job->get_input_tensors().empty());
  EXPECT_FALSE(master_job->get_input_memory_holders().empty());
}

TEST_F(StarPUTaskRunnerFixture, ReleasePendingJobsNoopsWhenNoAdditionalJobs)
{
  auto master_job = make_job(
      50, {torch::tensor({7.0F}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto master_holder = std::make_shared<int>(23);
  master_job->set_input_memory_holders(
      {std::shared_ptr<const void>(master_holder, master_holder.get())});

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> pending_jobs;
  starpu_server::StarPUTaskRunnerTestAdapter::release_pending_jobs(
      master_job, pending_jobs);

  EXPECT_TRUE(pending_jobs.empty());
  EXPECT_FALSE(master_job->get_input_tensors().empty());
  EXPECT_FALSE(master_job->get_input_memory_holders().empty());
}

TEST(SlotManagerCopyJobInputsToSlotTest, ReturnsImmediatelyWhenJobNull)
{
  std::shared_ptr<starpu_server::InferenceJob> missing_job;

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> pending_jobs;
  std::vector<starpu_data_handle_t> handles;
  std::vector<std::byte*> base_ptrs;
  std::vector<starpu_server::InputSlotPool::HostBufferInfo> buffer_infos;

  const std::span<const std::shared_ptr<starpu_server::InferenceJob>>
      pending_span(pending_jobs);
  const std::span<const starpu_data_handle_t> handle_span(handles);
  const std::span<std::byte* const> base_ptrs_span(base_ptrs);
  const std::span<const starpu_server::InputSlotPool::HostBufferInfo>
      buffer_info_span(buffer_infos);

  const auto bytes = test_api::slot_manager_copy_job_inputs_to_slot(
      missing_job, pending_span, handle_span, base_ptrs_span, buffer_info_span,
      nullptr);
  EXPECT_EQ(bytes, 0U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ConfigureTaskContextThrowsWhenOutputBytesMisaligned)
{
  auto model_config = make_model_config(
      "output_only", {}, {make_tensor_config("output0", {3}, at::kFloat)});

  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_output_pool());

  auto job = make_job(25, {});
  job->set_output_tensors(
      {torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat))});

  starpu_server::InferenceTask task(
      starpu_setup_.get(), job, &model_cpu_, &models_gpu_, &opts_,
      dependencies_);

  auto& output_pool = starpu_setup_->output_pool();
  const int slot = output_pool.acquire();
  const auto& output_handles = output_pool.handles(slot);
  ASSERT_EQ(output_handles.size(), 1U);
  const auto handle = output_handles[0];
  ASSERT_NE(handle, nullptr);

  const auto snapshots = snapshot_vector_interfaces(handle);
  ASSERT_FALSE(snapshots.empty());
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface != nullptr) {
      snapshot.iface->elemsize = snapshot.elemsize + 1;
    }
  }

  std::vector<starpu_data_handle_t> input_handles;
  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::configure_task_context(
          task, nullptr, -1, &output_pool, slot, input_handles, output_handles,
          /*batch_size=*/1),
      starpu_server::InvalidInferenceJobException);

  restore_vector_interfaces(snapshots);
  output_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ConfigureTaskContextThrowsWhenOutputCapacityExceeded)
{
  auto model_config = make_model_config(
      "output_only", {}, {make_tensor_config("output0", {3}, at::kFloat)});

  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_output_pool());

  auto job = make_job(26, {});
  job->set_output_tensors(
      {torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat))});

  starpu_server::InferenceTask task(
      starpu_setup_.get(), job, &model_cpu_, &models_gpu_, &opts_,
      dependencies_);

  auto& output_pool = starpu_setup_->output_pool();
  const int slot = output_pool.acquire();
  const auto& output_handles = output_pool.handles(slot);
  ASSERT_EQ(output_handles.size(), 1U);
  const auto handle = output_handles[0];
  ASSERT_NE(handle, nullptr);

  const auto snapshots = snapshot_vector_interfaces(handle);
  ASSERT_FALSE(snapshots.empty());
  const auto tensor_bytes =
      static_cast<std::size_t>(job->get_output_tensors()[0].nbytes());
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface != nullptr) {
      snapshot.iface->allocsize = tensor_bytes - 1;
    }
  }

  std::vector<starpu_data_handle_t> input_handles;
  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::configure_task_context(
          task, nullptr, -1, &output_pool, slot, input_handles, output_handles,
          /*batch_size=*/1),
      starpu_server::InvalidInferenceJobException);

  restore_vector_interfaces(snapshots);
  output_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ConfigureTaskContextThrowsWhenOutputHandleCountMismatch)
{
  auto model_config = make_model_config(
      "single_output_model", {},
      {make_tensor_config("output0", {3}, at::kFloat)});

  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_output_pool());

  const auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat);
  auto job = make_job(27, {});
  job->set_output_tensors(
      {torch::zeros({3}, tensor_opts), torch::zeros({3}, tensor_opts)});

  starpu_server::InferenceTask task(
      starpu_setup_.get(), job, &model_cpu_, &models_gpu_, &opts_,
      dependencies_);

  auto& output_pool = starpu_setup_->output_pool();
  const int slot = output_pool.acquire();
  const auto& output_handles = output_pool.handles(slot);
  ASSERT_EQ(output_handles.size(), 1U);

  std::vector<starpu_data_handle_t> input_handles;
  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::configure_task_context(
          task, nullptr, -1, &output_pool, slot, input_handles, output_handles,
          /*batch_size=*/1),
      starpu_server::InvalidInferenceJobException);

  output_pool.release(slot);
}

TEST_F(StarPUTaskRunnerFixture, ConfigureTaskContextThrowsWhenOutputHandleNull)
{
  auto model_config = make_model_config(
      "null_output_handle_model", {},
      {make_tensor_config("output0", {3}, at::kFloat)});

  reset_runner_with_model(model_config, /*pool_size=*/1);

  auto job = make_job(30, {});
  job->set_output_tensors(
      {torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat))});

  starpu_server::InferenceTask task(
      starpu_setup_.get(), job, &model_cpu_, &models_gpu_, &opts_,
      dependencies_);

  std::vector<starpu_data_handle_t> input_handles;
  std::vector<starpu_data_handle_t> output_handles = {nullptr};

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::configure_task_context(
          task, nullptr, -1, nullptr, -1, input_handles, output_handles,
          /*batch_size=*/1),
      starpu_server::StarPUDataAcquireException);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ConfigureTaskContextThrowsWhenHandleNotVectorInterface)
{
  auto model_config = make_model_config(
      "non_vector_handle_model", {},
      {make_tensor_config("output0", {2}, at::kFloat)});

  reset_runner_with_model(model_config, /*pool_size=*/1);

  auto job = make_job(31, {});
  job->set_output_tensors(
      {torch::zeros({2}, torch::TensorOptions().dtype(torch::kFloat))});

  starpu_server::InferenceTask task(
      starpu_setup_.get(), job, &model_cpu_, &models_gpu_, &opts_,
      dependencies_);

  int non_vector_value = 0;
  starpu_data_handle_t non_vector_handle = nullptr;
  starpu_variable_data_register(
      &non_vector_handle, STARPU_MAIN_RAM,
      reinterpret_cast<uintptr_t>(&non_vector_value), sizeof(non_vector_value));
  ASSERT_NE(non_vector_handle, nullptr);

  std::vector<starpu_data_handle_t> input_handles;
  std::vector<starpu_data_handle_t> output_handles = {non_vector_handle};
  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::configure_task_context(
          task, nullptr, -1, nullptr, -1, input_handles, output_handles,
          /*batch_size=*/1),
      starpu_server::StarPUDataAcquireException);

  starpu_data_unregister(non_vector_handle);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ConfigureTaskContextSkipsMissingVectorInterfacesOnNodes)
{
  auto model_config = make_model_config(
      "missing_interface_model", {},
      {make_tensor_config("output0", {2}, at::kFloat)});

  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_output_pool());

  auto job = make_job(32, {});
  job->set_output_tensors(
      {torch::zeros({2}, torch::TensorOptions().dtype(torch::kFloat))});

  starpu_server::InferenceTask task(
      starpu_setup_.get(), job, &model_cpu_, &models_gpu_, &opts_,
      dependencies_);

  auto& output_pool = starpu_setup_->output_pool();
  const int slot = output_pool.acquire();
  const auto& output_handles = output_pool.handles(slot);
  ASSERT_FALSE(output_handles.empty());

  missing_interface_override_handle = output_handles[0];
  missing_interface_override_hits = 0;

  std::vector<starpu_data_handle_t> input_handles;
  {
    starpu_test::ScopedStarpuMemoryNodesGetCountOverride nodes_override(
        two_memory_nodes_override);
    starpu_test::ScopedStarpuDataGetInterfaceOnNodeOverride interface_override(
        missing_interface_override);

    EXPECT_NO_THROW(
        starpu_server::StarPUTaskRunnerTestAdapter::configure_task_context(
            task, nullptr, -1, &output_pool, slot, input_handles,
            output_handles, /*batch_size=*/1));
  }

  EXPECT_GT(missing_interface_override_hits.load(), 0);

  missing_interface_override_handle = nullptr;
  output_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture, ConfigureTaskContextAllowsUndefinedOutputTensors)
{
  auto model_config = make_model_config(
      "undefined_output_model", {},
      {make_tensor_config("output0", {2}, at::kFloat)});

  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_output_pool());

  auto job = make_job(28, {});
  job->set_output_tensors({torch::Tensor()});

  starpu_server::InferenceTask task(
      starpu_setup_.get(), job, &model_cpu_, &models_gpu_, &opts_,
      dependencies_);

  auto& output_pool = starpu_setup_->output_pool();
  const int slot = output_pool.acquire();
  const auto& output_handles = output_pool.handles(slot);
  ASSERT_EQ(output_handles.size(), 1U);

  std::vector<starpu_data_handle_t> input_handles;
  EXPECT_NO_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::configure_task_context(
          task, nullptr, -1, &output_pool, slot, input_handles, output_handles,
          /*batch_size=*/1));

  output_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture, ConfigureTaskContextThrowsWhenOutputTensorNotCpu)
{
  auto model_config = make_model_config(
      "non_contiguous_output_model", {},
      {make_tensor_config("output0", {2, 2}, at::kFloat)});

  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_output_pool());

  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA device not available for non-CPU tensor test";
  }
  const auto tensor_opts =
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
  (void)cudaGetLastError();
  torch::Tensor gpu_tensor;
  try {
    gpu_tensor = torch::ones({2, 2}, tensor_opts);
  }
  catch (const c10::Error& e) {
    (void)cudaGetLastError();
    GTEST_SKIP() << "Unable to allocate CUDA tensor for non-CPU output test: "
                 << e.what();
  }
  EXPECT_FALSE(gpu_tensor.is_cpu());

  auto job = make_job(29, {});
  job->set_output_tensors({gpu_tensor});

  starpu_server::InferenceTask task(
      starpu_setup_.get(), job, &model_cpu_, &models_gpu_, &opts_,
      dependencies_);

  auto& output_pool = starpu_setup_->output_pool();
  const int slot = output_pool.acquire();
  const auto& output_handles = output_pool.handles(slot);
  ASSERT_EQ(output_handles.size(), 1U);

  std::vector<starpu_data_handle_t> input_handles;
  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::configure_task_context(
          task, nullptr, -1, &output_pool, slot, input_handles, output_handles,
          /*batch_size=*/1),
      starpu_server::InvalidInferenceJobException);

  output_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    HandleSubmissionFailureReleasesSlotsThroughTestHook)
{
  auto model_config = make_model_config(
      "test", {make_tensor_config("input0", {3}, at::kFloat)},
      {make_tensor_config("output0", {3}, at::kFloat)});

  reset_runner_with_model(model_config, /*pool_size=*/1);

  auto& input_pool = starpu_setup_->input_pool();
  auto& output_pool = starpu_setup_->output_pool();

  const int input_slot = input_pool.acquire();
  const int output_slot = output_pool.acquire();

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::handle_submission_failure(
          &input_pool, input_slot, &output_pool, output_slot, nullptr, -1),
      starpu_server::StarPUTaskSubmissionException);

  auto reacquired_input = input_pool.try_acquire();
  ASSERT_TRUE(reacquired_input.has_value());
  EXPECT_EQ(*reacquired_input, input_slot);
  input_pool.release(*reacquired_input);

  auto reacquired_output = output_pool.try_acquire();
  ASSERT_TRUE(reacquired_output.has_value());
  EXPECT_EQ(*reacquired_output, output_slot);
  output_pool.release(*reacquired_output);
}
