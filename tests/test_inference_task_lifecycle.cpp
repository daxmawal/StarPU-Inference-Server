#include <gtest/gtest.h>
#include <malloc.h>
#include <starpu.h>
#include <torch/script.h>

#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"

using namespace starpu_server;

namespace {
int acquire_cb_calls = 0;
int release_calls = 0;
bool job_callback_called = false;
bool malloc_fail_flag = false;
}  // namespace

extern "C" void* __libc_malloc(size_t);
extern "C" void __libc_free(void*);

extern "C" void*
malloc(size_t size)
{
  if (malloc_fail_flag) {
    malloc_fail_flag = false;
    return nullptr;
  }
  return __libc_malloc(size);
}

extern "C" void
free(void* ptr)
{
  __libc_free(ptr);
}

extern "C" void
starpu_data_release(starpu_data_handle_t)
{
  ++release_calls;
}

// ----------------------------------------------------------------------------
// Helper to create a minimal InferenceTask
// ----------------------------------------------------------------------------
static auto
make_task(
    StarPUSetup* starpu, std::shared_ptr<InferenceJob> job,
    RuntimeConfig* opts) -> std::unique_ptr<InferenceTask>
{
  static torch::jit::script::Module dummy_module{"m"};
  static std::vector<torch::jit::script::Module> dummy_models;
  return std::make_unique<InferenceTask>(
      starpu, std::move(job), &dummy_module, &dummy_models, opts);
}

// ----------------------------------------------------------------------------

TEST(InferenceTaskBuffers, FillTaskBuffersOrdersDynHandlesAndModes)
{
  auto ctx = std::make_shared<InferenceCallbackContext>(
      nullptr, nullptr, nullptr, 0, std::vector<starpu_data_handle_t>{},
      std::vector<starpu_data_handle_t>{});
  starpu_task* task = starpu_task_create();
  InferenceTask::allocate_task_buffers(task, 3, ctx);

  starpu_data_handle_t h1 = reinterpret_cast<starpu_data_handle_t>(0x1);
  starpu_data_handle_t h2 = reinterpret_cast<starpu_data_handle_t>(0x2);
  starpu_data_handle_t h3 = reinterpret_cast<starpu_data_handle_t>(0x3);

  InferenceTask::fill_task_buffers(task, {h1, h2}, {h3});

  EXPECT_EQ(task->dyn_handles[0], h1);
  EXPECT_EQ(task->dyn_handles[1], h2);
  EXPECT_EQ(task->dyn_handles[2], h3);
  EXPECT_EQ(task->dyn_modes[0], STARPU_R);
  EXPECT_EQ(task->dyn_modes[1], STARPU_R);
  EXPECT_EQ(task->dyn_modes[2], STARPU_W);
}
