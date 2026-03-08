#include <gtest/gtest.h>

#include "grpc/server/inference_service_test_api.hpp"

namespace {

TEST(UnaryCallDataHandleRequest, ReturnsInternalErrorWhenHandlerMissing)
{
  EXPECT_TRUE(starpu_server::testing::InferenceServiceTestAccessor::
                  UnaryCallDataMissingHandlerTransitionsToFinishForTest());
}

}  // namespace

namespace starpu_server { namespace {

TEST(ValidateConfiguredShape, RejectsNonPositiveBatchSize)
{
  const std::vector<int64_t> expected_shape = {2, 2};
  const std::vector<int64_t> incoming_shape = {0, 2, 2};

  grpc::Status status =
      testing::InferenceServiceTestAccessor::ValidateConfiguredShapeForTest(
          incoming_shape, expected_shape, /*batching_allowed=*/true,
          /*max_batch_size=*/4);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(
      status.error_message().find("batch size must be positive"),
      std::string::npos);
}

TEST(ValidateConfiguredShape, RejectsBatchSizeExceedingMaxLimit)
{
  const std::vector<int64_t> expected_shape = {2, 2};
  const std::vector<int64_t> incoming_shape = {5, 2, 2};

  grpc::Status status =
      testing::InferenceServiceTestAccessor::ValidateConfiguredShapeForTest(
          incoming_shape, expected_shape, /*batching_allowed=*/true,
          /*max_batch_size=*/4);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(
      status.error_message().find("exceeds configured max"), std::string::npos);
}

TEST(ValidateConfiguredShape, RejectsNonPositiveBatchSizeWhenConfiguredBatchDim)
{
  const std::vector<int64_t> expected_shape = {4, 2, 2};
  const std::vector<int64_t> incoming_shape = {0, 2, 2};

  grpc::Status status =
      testing::InferenceServiceTestAccessor::ValidateConfiguredShapeForTest(
          incoming_shape, expected_shape, /*batching_allowed=*/true,
          /*max_batch_size=*/4);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(
      status.error_message().find("batch size must be positive"),
      std::string::npos);
}

TEST(ValidateConfiguredShape, RejectsOversizedBatchWhenConfiguredBatchDim)
{
  const std::vector<int64_t> expected_shape = {4, 2, 2};
  const std::vector<int64_t> incoming_shape = {5, 2, 2};

  grpc::Status status =
      testing::InferenceServiceTestAccessor::ValidateConfiguredShapeForTest(
          incoming_shape, expected_shape, /*batching_allowed=*/true,
          /*max_batch_size=*/4);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(
      status.error_message().find("exceeds configured max"), std::string::npos);
}

}}  // namespace starpu_server
