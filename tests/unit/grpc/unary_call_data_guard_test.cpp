#include <gtest/gtest.h>

#define private public
#define protected public
#include <grpcpp/impl/call.h>
#include <grpcpp/impl/call_hook.h>
#include <grpcpp/impl/call_op_set_interface.h>

#include "../../../src/grpc/server/inference_service.cpp"
#undef protected
#undef private

namespace {

using ServerLiveRequest = inference::ServerLiveRequest;
using ServerLiveResponse = inference::ServerLiveResponse;

using UnaryServerLiveCallData =
    starpu_server::UnaryCallData<ServerLiveRequest, ServerLiveResponse>;

struct NoopCallHook : grpc::internal::CallHook {
  void PerformOpsOnCall(
      grpc::internal::CallOpSetInterface*, grpc::internal::Call*) override
  {
  }
};

TEST(UnaryCallDataHandleRequest, ReturnsInternalErrorWhenHandlerMissing)
{
  UnaryServerLiveCallData call_data(
      /*service=*/nullptr,
      /*completion_queue=*/nullptr,
      /*impl=*/nullptr,
      /*request_method=*/
      UnaryServerLiveCallData::RequestMethod{},
      /*handler=*/UnaryServerLiveCallData::Handler{});

  ASSERT_FALSE(call_data.handler_);

  NoopCallHook noop_hook{};
  call_data.responder_.call_.call_hook_ = &noop_hook;

  call_data.HandleRequest();

  EXPECT_EQ(call_data.status_, UnaryServerLiveCallData::CallStatus::Finish);
}

}  // namespace

namespace starpu_server { namespace {

TEST(ValidateConfiguredShape, RejectsNonPositiveBatchSize)
{
  const std::vector<int64_t> expected_shape = {2, 2};
  const std::vector<int64_t> incoming_shape = {0, 2, 2};

  grpc::Status status = validate_configured_shape(
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

  grpc::Status status = validate_configured_shape(
      incoming_shape, expected_shape, /*batching_allowed=*/true,
      /*max_batch_size=*/4);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(
      status.error_message().find("exceeds configured max"), std::string::npos);
}

}}  // namespace starpu_server
