#include <gtest/gtest.h>

#define private public
#include "grpc/server/inference_service.hpp"
#undef private

namespace {

using CallbackHandle = starpu_server::InferenceServiceImpl::CallbackHandle;

TEST(CallbackHandleTest, InvokeDoesNothingWhenCallbackMissing)
{
  int invocation_count = 0;
  grpc::Status last_status = grpc::Status(
      grpc::StatusCode::UNKNOWN, "uninitialized sentinel for the test");

  CallbackHandle handle([&](grpc::Status status) {
    ++invocation_count;
    last_status = std::move(status);
  });

  handle.Invoke(grpc::Status::OK);
  EXPECT_EQ(invocation_count, 1);
  EXPECT_TRUE(last_status.ok());

  // Second invoke must early-exit because the callback has already been moved.
  handle.Invoke(grpc::Status(
      grpc::StatusCode::CANCELLED, "should not be seen by the callback"));
  EXPECT_EQ(invocation_count, 1);
  EXPECT_TRUE(last_status.ok());
}

}  // namespace
