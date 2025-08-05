#include <gtest/gtest.h>
#include <malloc.h>
#include <starpu.h>
#include <torch/script.h>

#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"

TEST(InferenceTaskBuffers, FillTaskBuffersOrdersDynHandlesAndModes)
{
  auto ctx = std::make_shared<starpu_server::InferenceCallbackContext>(
      nullptr, nullptr, nullptr, 0, std::vector<starpu_data_handle_t>{},
      std::vector<starpu_data_handle_t>{});
  starpu_task* task = starpu_task_create();
  starpu_server::InferenceTask::allocate_task_buffers(task, 3, ctx);
  starpu_data_handle_t h1 = reinterpret_cast<starpu_data_handle_t>(0x1);
  starpu_data_handle_t h2 = reinterpret_cast<starpu_data_handle_t>(0x2);
  starpu_data_handle_t h3 = reinterpret_cast<starpu_data_handle_t>(0x3);
  starpu_server::InferenceTask::fill_task_buffers(task, {h1, h2}, {h3});
  EXPECT_EQ(task->dyn_handles[0], h1);
  EXPECT_EQ(task->dyn_handles[1], h2);
  EXPECT_EQ(task->dyn_handles[2], h3);
  EXPECT_EQ(task->dyn_modes[0], STARPU_R);
  EXPECT_EQ(task->dyn_modes[1], STARPU_R);
  EXPECT_EQ(task->dyn_modes[2], STARPU_W);
}
