#include <gtest/gtest.h>

#include <vector>

#include "cli/args_parser.hpp"
#include "test_cli.hpp"
#include "utils/exceptions.hpp"

static auto build_common_args() -> std::vector<const char*> {
  return {"program", "--model", test_model_path().c_str(), "--shape", "1x3",
          "--types", "float"};
}

class ArgsParserInvalidOptions_Robustesse
    : public ::testing::TestWithParam<std::vector<const char*>> {};

TEST_P(ArgsParserInvalidOptions_Robustesse, Invalid)
{
  auto args = build_common_args();
  const auto& diff = GetParam();
  args.insert(args.end(), diff.begin(), diff.end());
  expect_invalid(args);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidArguments, ArgsParserInvalidOptions_Robustesse,
    ::testing::Values(
        std::vector<const char*>{"--types", ""},
        std::vector<const char*>{"--iterations", "0"},
        std::vector<const char*>{"--shape", "1x-3x3"},
        std::vector<const char*>{"--types", "unknown"},
        std::vector<const char*>{"--scheduler", "unknown"},
        std::vector<const char*>{"--device-ids", "-1"},
        std::vector<const char*>{"--device-ids", "0,0"},
        std::vector<const char*>{"--unknown"},
        std::vector<const char*>{"--shapes", "1x2,2x3", "--types", "float"},
        std::vector<const char*>{"--shape", "1xax2"},
        std::vector<const char*>{"--shape", "9223372036854775808"},
        std::vector<const char*>{"--verbose", "5"},
        std::vector<const char*>{"--shape", ""},
        std::vector<const char*>{"--shapes", ""},
        std::vector<const char*>{"--shapes", "1x2,", "--types", "float"},
        std::vector<const char*>{"--iterations", "-1"},
        std::vector<const char*>{"--delay", "-1"},
        std::vector<const char*>{"--max-batch-size", "0"},
        std::vector<const char*>{"--max-batch-size", "-1"},
        std::vector<const char*>{"--metrics-port", "0"},
        std::vector<const char*>{"--metrics-port", "65536"},
        std::vector<const char*>{"--pregen-inputs", "0"},
        std::vector<const char*>{"--pregen-inputs", "-1"},
        std::vector<const char*>{"--warmup-iterations", "-1"},
        std::vector<const char*>{"--seed", "-1"},
        std::vector<const char*>{"--shapes", "1x2,,3", "--types", "float,int"},
        std::vector<const char*>{"--model"},
        std::vector<const char*>{"--iterations"},
        std::vector<const char*>{"--model",
                                 "/nonexistent/path/to/model.pt"},
        std::vector<const char*>{"--config",
                                 "/nonexistent/path/to/config.yaml"}));
