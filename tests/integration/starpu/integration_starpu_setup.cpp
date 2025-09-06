#include "test_starpu_setup.hpp"

class StarPUSetupCodelet_Integration : public ::testing::Test {
 protected:
  auto starpu_setup() -> starpu_server::StarPUSetup& { return *starpu_; }
  [[nodiscard]] auto starpu_setup() const -> const starpu_server::StarPUSetup&
  {
    return *starpu_;
  }

  void SetUp() override
  {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP();
    }
    starpu_server::RuntimeConfig opts;
    opts.use_cpu = true;
    opts.use_cuda = true;
    opts.device_ids = {0};
    starpu_ = std::make_unique<starpu_server::StarPUSetup>(opts);
  }

 private:
  std::unique_ptr<starpu_server::StarPUSetup> starpu_;
};

TEST_F(StarPUSetupCodelet_Integration, GetCodeletNotNull)
{
  EXPECT_NE(starpu_setup().get_codelet(), nullptr);
}

TEST_F(StarPUSetupCodelet_Integration, GetCudaWorkersSingleDevice)
{
  auto workers = starpu_server::StarPUSetup::get_cuda_workers_by_device({0});
  EXPECT_FALSE(workers.empty());
}

TEST(InferenceCodelet_Integration, CpuInferenceFuncExecutesAndSetsMetadata)
{
  StarpuRuntimeGuard starpu_guard;
  auto buffers = make_test_buffers();
  auto timing = setup_timing_params(3);
  starpu_server::InferenceCodelet inf_cl;
  auto* codelet = inf_cl.get_codelet();
  auto before = std::chrono::high_resolution_clock::now();
  codelet->cpu_funcs[0](buffers.buffers.data(), &timing.params);
  auto after = std::chrono::high_resolution_clock::now();
  EXPECT_FLOAT_EQ(buffers.output_data[0], 2.0F);
  EXPECT_FLOAT_EQ(buffers.output_data[1], 3.0F);
  EXPECT_FLOAT_EQ(buffers.output_data[2], 4.0F);
  EXPECT_EQ(timing.executed_on, starpu_server::DeviceType::CPU);
  EXPECT_TRUE(timing.start_time >= before);
  EXPECT_TRUE(timing.end_time <= after);
  EXPECT_TRUE(timing.end_time >= timing.start_time);
}
