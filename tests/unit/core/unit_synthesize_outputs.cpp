#include <gtest/gtest.h>
#include <prometheus/client_metric.h>
#include <prometheus/metric_family.h>

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../../../src/core/inference_runner.cpp"
#include "monitoring/metrics.hpp"
#include "test_helpers.hpp"
#include "utils/runtime_config.hpp"

namespace starpu_server { namespace {

auto
make_valid_model_config() -> ModelConfig
{
  ModelConfig model;
  model.name = "test-model";
  model.outputs = {
      TensorConfig{
          .name = "output0",
          .dims = {2, 3},
          .type = at::kFloat,
      },
      TensorConfig{
          .name = "output1",
          .dims = {1, 4},
          .type = at::kDouble,
      }};
  return model;
}

auto
make_runtime_config_for_model(const std::filesystem::path& path)
    -> RuntimeConfig
{
  RuntimeConfig opts;
  ModelConfig model{};
  model.path = path.string();
  model.inputs = {TensorConfig{
      .name = "input0",
      .dims = {1},
      .type = at::kFloat,
  }};
  opts.model = std::move(model);
  opts.devices.use_cuda = false;
  return opts;
}

auto
FindFamily(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view name) -> const prometheus::MetricFamily*
{
  for (const auto& family : families) {
    if (family.name == name) {
      return &family;
    }
  }
  return nullptr;
}

auto
MetricMatchesLabels(
    const prometheus::ClientMetric& metric,
    const std::vector<std::pair<std::string_view, std::string_view>>& labels)
    -> bool
{
  for (const auto& [label_name, label_value] : labels) {
    bool matched = false;
    for (const auto& label : metric.label) {
      if (label.name == label_name && label.value == label_value) {
        matched = true;
        break;
      }
    }
    if (!matched) {
      return false;
    }
  }
  return true;
}

auto
FindGaugeValue(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view family_name,
    const std::vector<std::pair<std::string_view, std::string_view>>& labels)
    -> std::optional<double>
{
  const auto* family = FindFamily(families, family_name);
  if (family == nullptr) {
    return std::nullopt;
  }
  for (const auto& metric : family->metric) {
    if (MetricMatchesLabels(metric, labels)) {
      return metric.gauge.value;
    }
  }
  return std::nullopt;
}

auto
FindCounterValue(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view family_name,
    const std::vector<std::pair<std::string_view, std::string_view>>& labels)
    -> std::optional<double>
{
  const auto* family = FindFamily(families, family_name);
  if (family == nullptr) {
    return std::nullopt;
  }
  for (const auto& metric : family->metric) {
    if (MetricMatchesLabels(metric, labels)) {
      return metric.counter.value;
    }
  }
  return std::nullopt;
}

auto
FindHistogramSampleCount(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view family_name) -> std::optional<std::uint64_t>
{
  const auto* family = FindFamily(families, family_name);
  if (family == nullptr || family->metric.empty()) {
    return std::nullopt;
  }
  return family->metric.front().histogram.sample_count;
}

TEST(SynthesizeOutputsFromConfig, ReturnsNulloptWhenNoModels)
{
  RuntimeConfig opts;
  auto outputs = synthesize_outputs_from_config(opts);
  EXPECT_FALSE(outputs.has_value());
}

TEST(SynthesizeOutputsFromConfig, ReturnsNulloptWhenOutputsMissing)
{
  RuntimeConfig opts;
  ModelConfig model;
  opts.model = model;

  auto outputs = synthesize_outputs_from_config(opts);
  EXPECT_FALSE(outputs.has_value());
}

TEST(SynthesizeOutputsFromConfig, ReturnsNulloptWhenTypeMissing)
{
  RuntimeConfig opts;
  auto model = make_valid_model_config();
  model.outputs[0].type = at::ScalarType::Undefined;
  opts.model = model;

  CaptureStream capture{std::cerr};
  auto outputs = synthesize_outputs_from_config(opts);

  EXPECT_FALSE(outputs.has_value());
  EXPECT_NE(capture.str().find("missing a valid data_type"), std::string::npos);
}

TEST(SynthesizeOutputsFromConfig, ReturnsNulloptWhenDimsMissing)
{
  RuntimeConfig opts;
  auto model = make_valid_model_config();
  model.outputs[0].dims.clear();
  opts.model = model;

  CaptureStream capture{std::cerr};
  auto outputs = synthesize_outputs_from_config(opts);

  EXPECT_FALSE(outputs.has_value());
  EXPECT_NE(capture.str().find("missing dims"), std::string::npos);
}

TEST(SynthesizeOutputsFromConfig, ReturnsNulloptWhenDimNonPositive)
{
  RuntimeConfig opts;
  auto model = make_valid_model_config();
  model.outputs[0].dims[0] = 0;
  opts.model = model;

  CaptureStream capture{std::cerr};
  auto outputs = synthesize_outputs_from_config(opts);

  EXPECT_FALSE(outputs.has_value());
  EXPECT_NE(capture.str().find("non-positive dimension"), std::string::npos);
}

TEST(SynthesizeOutputsFromConfig, CreatesOutputsWhenConfigValid)
{
  RuntimeConfig opts;
  opts.model = make_valid_model_config();

  auto outputs = synthesize_outputs_from_config(opts);

  ASSERT_TRUE(outputs.has_value());
  ASSERT_EQ(outputs->size(), 2U);
  EXPECT_TRUE(outputs->at(0).sizes().vec() == opts.model->outputs[0].dims);
  EXPECT_EQ(outputs->at(0).dtype(), torch::kFloat32);
  EXPECT_TRUE(outputs->at(1).sizes().vec() == opts.model->outputs[1].dims);
  EXPECT_EQ(outputs->at(1).dtype(), torch::kFloat64);
}

TEST(LoadModelAndReferenceOutput, MetricsUseDefaultLabelOnFailure)
{
  shutdown_metrics();
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  RuntimeConfig opts;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0, 2};

  const auto result = load_model_and_reference_output(opts);

  EXPECT_FALSE(result.has_value());

  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto families = metrics->registry()->Collect();

  const auto failure_value = FindCounterValue(
      families, "model_load_failures_total", {{"model", "default"}});
  ASSERT_TRUE(failure_value.has_value());
  EXPECT_DOUBLE_EQ(*failure_value, 1.0);

  const auto cpu_loaded = FindGaugeValue(
      families, "models_loaded", {{"model", "default"}, {"device", "cpu"}});
  ASSERT_TRUE(cpu_loaded.has_value());
  EXPECT_DOUBLE_EQ(*cpu_loaded, 0.0);

  for (const auto device_id : opts.devices.ids) {
    const auto device_label = std::string("cuda:") + std::to_string(device_id);
    const auto gpu_loaded = FindGaugeValue(
        families, "models_loaded",
        {{"model", "default"}, {"device", device_label}});
    ASSERT_TRUE(gpu_loaded.has_value());
    EXPECT_DOUBLE_EQ(*gpu_loaded, 0.0);
  }
}

TEST(LoadModelAndReferenceOutput, MetricsUseModelNameWhenProvided)
{
  shutdown_metrics();
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  TemporaryModelFile model_file{"load_model_named", make_add_one_model()};
  RuntimeConfig opts = make_runtime_config_for_model(model_file.path());
  opts.model->name = "named-model";
  opts.model->outputs = {TensorConfig{
      .name = "output0",
      .dims = {1},
      .type = at::kFloat,
  }};

  const auto result = load_model_and_reference_output(opts);

  ASSERT_TRUE(result.has_value());

  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto families = metrics->registry()->Collect();

  const auto cpu_loaded = FindGaugeValue(
      families, "models_loaded", {{"model", "named-model"}, {"device", "cpu"}});
  ASSERT_TRUE(cpu_loaded.has_value());
  EXPECT_DOUBLE_EQ(*cpu_loaded, 1.0);
}

TEST(LoadModelAndReferenceOutput, MetricsSetGpuLoadedFlagsOnSuccess)
{
  skip_if_no_cuda();

  const auto device_count = detail::get_cuda_device_count();
  if (device_count < 1) {
    GTEST_SKIP() << "CUDA device count is zero.";
  }

  shutdown_metrics();
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  TemporaryModelFile model_file{"load_model_gpu", make_add_one_model()};
  RuntimeConfig opts = make_runtime_config_for_model(model_file.path());
  opts.model->name = "gpu-model";
  opts.model->outputs = {TensorConfig{
      .name = "output0",
      .dims = {1},
      .type = at::kFloat,
  }};
  opts.devices.use_cuda = true;
  opts.devices.ids = {0};

  const auto result = load_model_and_reference_output(opts);

  ASSERT_TRUE(result.has_value());

  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto families = metrics->registry()->Collect();

  const auto gpu_loaded = FindGaugeValue(
      families, "models_loaded",
      {{"model", "gpu-model"}, {"device", "cuda:0"}});
  ASSERT_TRUE(gpu_loaded.has_value());
  EXPECT_DOUBLE_EQ(*gpu_loaded, 1.0);
}

TEST(InferenceRunnerMetrics, SetModelLoadedMetricUsesRecorderWhenProvided)
{
  shutdown_metrics();
  auto recorder = create_metrics_recorder(0);
  ASSERT_NE(recorder, nullptr);
  ASSERT_TRUE(recorder->enabled());

  set_model_loaded_metric(
      recorder.get(), "recorder-model", "cuda:2", /*loaded=*/true);

  const auto families = recorder->registry()->registry()->Collect();
  const auto loaded = FindGaugeValue(
      families, "models_loaded",
      {{"model", "recorder-model"}, {"device", "cuda:2"}});
  ASSERT_TRUE(loaded.has_value());
  EXPECT_DOUBLE_EQ(*loaded, 1.0);

  shutdown_metrics();
}

TEST(
    InferenceRunnerMetrics,
    ModelLoadRecorderHelpersRecordFailureFlagsAndLoadDuration)
{
  shutdown_metrics();
  auto recorder = create_metrics_recorder(0);
  ASSERT_NE(recorder, nullptr);
  ASSERT_TRUE(recorder->enabled());

  RuntimeConfig opts;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0};

  mark_model_load_failure(opts, recorder.get(), "recorder-model");
  record_model_load_success(
      opts, recorder.get(), "recorder-model",
      MonotonicClock::now() - std::chrono::milliseconds(5));

  const auto families = recorder->registry()->registry()->Collect();

  const auto failure_value = FindCounterValue(
      families, "model_load_failures_total", {{"model", "recorder-model"}});
  ASSERT_TRUE(failure_value.has_value());
  EXPECT_DOUBLE_EQ(*failure_value, 1.0);

  const auto cpu_loaded = FindGaugeValue(
      families, "models_loaded",
      {{"model", "recorder-model"}, {"device", "cpu"}});
  ASSERT_TRUE(cpu_loaded.has_value());
  EXPECT_DOUBLE_EQ(*cpu_loaded, 1.0);

  const auto gpu_loaded = FindGaugeValue(
      families, "models_loaded",
      {{"model", "recorder-model"}, {"device", "cuda:0"}});
  ASSERT_TRUE(gpu_loaded.has_value());
  EXPECT_DOUBLE_EQ(*gpu_loaded, 1.0);

  const auto load_duration_samples =
      FindHistogramSampleCount(families, "model_load_duration_ms");
  ASSERT_TRUE(load_duration_samples.has_value());
  EXPECT_GE(*load_duration_samples, 1U);

  shutdown_metrics();
}

TEST(LoadModelAndReferenceOutput, LogsFallbackWhenSyntheticMissing)
{
  TemporaryModelFile model_file{"load_model_missing", make_add_one_model()};
  RuntimeConfig opts = make_runtime_config_for_model(model_file.path());
  opts.verbosity = VerbosityLevel::Debug;
  opts.model->outputs = {TensorConfig{
      .name = "bad_output",
      .dims = {},
      .type = at::kFloat,
  }};

  CaptureStream capture{std::cout};
  const auto result = load_model_and_reference_output(opts);

  EXPECT_TRUE(result.has_value());
  EXPECT_NE(
      capture.str().find("No usable output schema provided"),
      std::string::npos);
}

TEST(LoadModelAndReferenceOutput, LogsWhenUsingSyntheticOutputs)
{
  TemporaryModelFile model_file{"load_model_synthetic", make_add_one_model()};
  RuntimeConfig opts = make_runtime_config_for_model(model_file.path());
  opts.verbosity = VerbosityLevel::Debug;
  opts.model->outputs = {TensorConfig{
      .name = "output0",
      .dims = {1},
      .type = at::kFloat,
  }};

  CaptureStream capture{std::cout};
  const auto result = load_model_and_reference_output(opts);

  ASSERT_TRUE(result.has_value());
  EXPECT_NE(
      capture.str().find(
          "Using configured output schema instead of running CPU reference "
          "inference."),
      std::string::npos);
  const auto& outputs = std::get<2>(*result);
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_TRUE(outputs[0].defined());
  EXPECT_EQ(outputs[0].sizes().vec(), opts.model->outputs[0].dims);
}

TEST(InferenceRunner, LoadModelLoadsTorchScriptModule)
{
  TemporaryModelFile model_file{"load_model_basic", make_add_one_model()};

  auto module = load_model(model_file.path().string());

  const auto input = torch::ones({1}, torch::TensorOptions().dtype(at::kFloat));
  const auto output = module.forward({input}).toTensor();
  EXPECT_TRUE(output.allclose(input + 1));
}

TEST(InferenceRunner, CloneModelToGpusReturnsEmptyWhenNoDeviceIds)
{
  auto cpu_model = make_add_one_model();
  starpu_server::RuntimeConfig opts;
  opts.devices.use_cuda = true;

  const auto gpu_models = clone_model_to_gpus(cpu_model, opts);

  EXPECT_TRUE(gpu_models.empty());
}

}}  // namespace starpu_server
