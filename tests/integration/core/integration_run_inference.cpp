#include <gtest/gtest.h>

TEST(RunInference_Robustesse, NoRobustnessScenariosYet)
{
  GTEST_SKIP()
      << "No error behavior defined for run_inference()."
         "Add tests here (null buffers, inconsistent sizes, null model)"
         "as soon as the API specifies what to throw/return.";
}
