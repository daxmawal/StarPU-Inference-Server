#pragma once

#include <starpu.h>
#include <torch/script.h>

#include <ostream>
#include <sstream>
#include <vector>

#include "core/inference_params.hpp"
#include "grpc_service.grpc.pb.h"

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

inline starpu_variable_interface
make_variable_interface(float* ptr)
{
  starpu_variable_interface iface;
  iface.ptr = reinterpret_cast<uintptr_t>(ptr);
  return iface;
}

inline InferenceParams
make_basic_params(int elements, at::ScalarType type = at::kFloat)
{
  InferenceParams params{};
  params.num_inputs = 1;
  params.num_outputs = 1;
  params.layout.num_dims[0] = 1;
  params.layout.dims[0][0] = elements;
  params.layout.input_types[0] = type;
  return params;
}

inline inference::ModelInferRequest
make_valid_request()
{
  inference::ModelInferRequest req;
  auto* input = req.add_inputs();
  input->set_name("input0");
  input->set_datatype("FP32");
  input->add_shape(2);
  input->add_shape(2);

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  req.add_raw_input_contents()->assign(
      reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
  return req;
}

}  // namespace starpu_server
