#include <gtest/gtest.h>

TEST(InferenceRunnerHelpers_Integration, NoIntegrationScenarioYet)
{
  GTEST_SKIP() << "No disk integration test (yet): "
                  "add a happy path case that serializes a TorchScript Module, "
                  "then calls load_model_and_reference_output().";
}
