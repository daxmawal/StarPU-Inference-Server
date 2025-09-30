#pragma once

#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include <bit>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <format>
#include <functional>
#include <future>
#include <ostream>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core/inference_params.hpp"
#include "grpc/server/inference_service.hpp"
#include "grpc_service.grpc.pb.h"
#include "starpu_task_worker/inference_queue.hpp"
#include "utils/datatype_utils.hpp"
#include "utils/logger.hpp"

inline void
skip_if_no_cuda()
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
}

namespace starpu_server {

class CaptureStream {
 public:
  explicit CaptureStream(std::ostream& stream)
      : stream_{stream}, old_buf_{stream.rdbuf(buffer_.rdbuf())}
  {
  }
  ~CaptureStream() { stream_.rdbuf(old_buf_); }
  [[nodiscard]] auto str() const -> std::string { return buffer_.str(); }

 private:
  std::ostream& stream_;
  std::ostringstream buffer_;
  std::streambuf* old_buf_;
};

inline constexpr VerbosityLevel WarningLevel = static_cast<VerbosityLevel>(100);
inline constexpr VerbosityLevel ErrorLevel = static_cast<VerbosityLevel>(101);

inline auto
expected_log_line(VerbosityLevel level, const std::string& msg) -> std::string
{
  if (level == WarningLevel) {
    return std::string{"\x1b[1;33m[WARNING] "} + msg + "\x1b[0m\n";
  }
  if (level == ErrorLevel) {
    return std::string{"\x1b[1;31m[ERROR] "} + msg + "\x1b[0m\n";
  }
  auto [color, label] = verbosity_style(level);
  return std::string(color) + label + msg + "\x1b[0m\n";
}

inline auto
make_add_one_model() -> torch::jit::script::Module
{
  torch::jit::script::Module module{"m"};
  module.define(R"JIT(
        def forward(self, x):
            return x + 1
    )JIT");
  return module;
}

inline auto
make_variable_interface(const float* ptr) -> starpu_variable_interface
{
  starpu_variable_interface iface{};
  iface.ptr = std::bit_cast<uintptr_t>(ptr);
  return iface;
}

template <typename T>
inline auto
make_variable_interface(const T* ptr) -> starpu_variable_interface
{
  starpu_variable_interface iface{};
  iface.ptr = std::bit_cast<uintptr_t>(ptr);
  return iface;
}

inline auto
make_basic_params(int elements, at::ScalarType type = at::kFloat)
    -> InferenceParams
{
  InferenceParams params{};
  params.num_inputs = 1;
  params.num_outputs = 1;
  params.limits.max_inputs = InferLimits::MaxInputs;
  params.limits.max_dims = InferLimits::MaxDims;
  params.limits.max_models_gpu = InferLimits::MaxModelsGPU;
  params.layout.num_dims.resize(1);
  params.layout.num_dims[0] = 1;
  params.layout.dims.resize(1);
  params.layout.dims[0] = {elements};
  params.layout.input_types.resize(1);
  params.layout.input_types[0] = type;
  return params;
}

inline auto
make_params_for_inputs(
    const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<at::ScalarType>& dtypes) -> InferenceParams
{
  assert(shapes.size() == dtypes.size());
  InferenceParams params{};
  params.num_inputs = shapes.size();
  params.limits.max_inputs = InferLimits::MaxInputs;
  params.limits.max_dims = InferLimits::MaxDims;
  params.limits.max_models_gpu = InferLimits::MaxModelsGPU;
  params.layout.num_dims.resize(shapes.size());
  params.layout.dims.resize(shapes.size());
  params.layout.input_types.resize(shapes.size());
  for (size_t i = 0; i < shapes.size(); ++i) {
    params.layout.num_dims[i] = static_cast<int64_t>(shapes[i].size());
    params.layout.dims[i] = shapes[i];
    params.layout.input_types[i] = dtypes[i];
  }
  return params;
}

struct InputSpec {
  std::vector<int64_t> shape;
  at::ScalarType dtype;
  std::string raw_data;
};

template <typename T>
auto
to_raw_data(const std::vector<T>& values) -> std::string
{
  auto byte_span = std::as_bytes(std::span(values));
  return std::string(
      reinterpret_cast<const char*>(byte_span.data()), byte_span.size());
}


inline auto
make_model_infer_request(const std::vector<InputSpec>& specs)
    -> inference::ModelInferRequest
{
  inference::ModelInferRequest req;
  for (size_t i = 0; i < specs.size(); ++i) {
    const auto& spec = specs[i];
    auto* input = req.add_inputs();
    input->set_name(std::format("input{}", i));
    input->set_datatype(scalar_type_to_datatype(spec.dtype));
    for (auto dim : spec.shape) {
      input->add_shape(dim);
    }
    req.add_raw_input_contents()->assign(spec.raw_data);
  }
  return req;
}

inline const InputSpec kValidInputSpec{
    {2, 2}, at::kFloat, to_raw_data<float>({1.0F, 2.0F, 3.0F, 4.0F})};

inline auto
make_valid_request() -> inference::ModelInferRequest
{
  return make_model_infer_request({kValidInputSpec});
}

inline auto
make_model_request(const std::string& name, const std::string& version)
    -> inference::ModelInferRequest
{
  inference::ModelInferRequest req;
  req.set_model_name(name);
  req.set_model_version(version);
  return req;
}

inline auto
run_single_job(
    InferenceQueue& queue, std::vector<torch::Tensor> outputs = {},
    double latency = 0.0,
    std::function<void(InferenceJob&)> job_mutator = {}) -> std::jthread
{
  return std::jthread([&queue, outputs = std::move(outputs), latency,
                       job_mutator = std::move(job_mutator)]() mutable {
    std::shared_ptr<InferenceJob> job;
    if (queue.wait_and_pop(job)) {
      if (job_mutator) {
        job_mutator(*job);
      }
      job->get_on_complete()(outputs, latency);
    }
  });
}

struct TestGrpcServer {
  std::unique_ptr<grpc::Server> server;
  std::jthread thread;
  int port;
};

inline auto
start_test_grpc_server(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    std::vector<at::ScalarType> expected_input_types = {kValidInputSpec.dtype},
    int port = 0) -> TestGrpcServer
{
  TestGrpcServer handle;
  std::promise<int> port_promise;
  auto port_future = port_promise.get_future();
  handle.thread = std::jthread([&queue, &reference_outputs, port, &handle,
                                expected_input_types,
                                p = std::move(port_promise)]() mutable {
    InferenceServiceImpl service(
        &queue, &reference_outputs, expected_input_types);
    grpc::ServerBuilder builder;
    std::string address = std::format("0.0.0.0:{}", port);
    int selected_port = 0;
    builder.AddListeningPort(
        address, grpc::InsecureServerCredentials(), &selected_port);
    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(32 * 1024 * 1024);
    builder.SetMaxSendMessageSize(32 * 1024 * 1024);
    handle.server = builder.BuildAndStart();
    p.set_value(selected_port);
    handle.server->Wait();
    handle.server.reset();
  });
  handle.port = port_future.get();
  while (!handle.server) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return handle;
}

inline void
verify_populate_response(
    const inference::ModelInferRequest& req,
    const inference::ModelInferResponse& resp,
    const std::vector<torch::Tensor>& outputs, uint64_t recv_ms,
    uint64_t send_ms,
    const starpu_server::InferenceServiceImpl::LatencyBreakdown& breakdown)
{
  EXPECT_EQ(resp.model_name(), req.model_name());
  EXPECT_EQ(resp.model_version(), req.model_version());
  EXPECT_EQ(resp.server_receive_ms(), recv_ms);
  EXPECT_EQ(resp.server_send_ms(), send_ms);
  EXPECT_DOUBLE_EQ(resp.server_preprocess_ms(), breakdown.preprocess_ms);
  EXPECT_DOUBLE_EQ(resp.server_queue_ms(), breakdown.queue_ms);
  EXPECT_DOUBLE_EQ(resp.server_submit_ms(), breakdown.submit_ms);
  EXPECT_DOUBLE_EQ(resp.server_scheduling_ms(), breakdown.scheduling_ms);
  EXPECT_DOUBLE_EQ(resp.server_codelet_ms(), breakdown.codelet_ms);
  EXPECT_DOUBLE_EQ(resp.server_inference_ms(), breakdown.inference_ms);
  EXPECT_DOUBLE_EQ(resp.server_callback_ms(), breakdown.callback_ms);
  EXPECT_DOUBLE_EQ(resp.server_postprocess_ms(), breakdown.postprocess_ms);
  EXPECT_DOUBLE_EQ(resp.server_total_ms(), breakdown.total_ms);
  EXPECT_DOUBLE_EQ(resp.server_overall_ms(), breakdown.overall_ms);

  ASSERT_EQ(resp.outputs_size(), static_cast<int>(outputs.size()));
  ASSERT_EQ(resp.raw_output_contents_size(), static_cast<int>(outputs.size()));

  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& out = resp.outputs(static_cast<int>(i));
    EXPECT_EQ(out.name(), "output" + std::to_string(i));
    EXPECT_EQ(
        out.datatype(), scalar_type_to_datatype(outputs[i].scalar_type()));
    ASSERT_EQ(out.shape_size(), outputs[i].dim());
    for (int64_t j = 0; j < outputs[i].dim(); ++j) {
      EXPECT_EQ(out.shape(j), outputs[i].size(j));
    }

    auto flat = outputs[i].view({-1});
    const auto& raw = resp.raw_output_contents(static_cast<int>(i));
    ASSERT_EQ(raw.size(), flat.numel() * flat.element_size());
    EXPECT_EQ(0, std::memcmp(raw.data(), flat.data_ptr(), raw.size()));
  }
}

}  // namespace starpu_server
