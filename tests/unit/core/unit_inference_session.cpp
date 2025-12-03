#include <gtest/gtest.h>

#include <stdexcept>

#define private public
#include "core/inference_session.hpp"
#undef private

#include "core/starpu_setup.hpp"
#include "utils/runtime_config.hpp"

namespace starpu_server {

TEST(InferenceSession, LaunchThreadsThrowsWhenWorkerMissing)
{
  RuntimeConfig opts;
  StarPUSetup starpu(opts);
  InferenceSession session(opts, starpu);

  EXPECT_THROW(session.launch_threads(), std::logic_error);
}

}  // namespace starpu_server
