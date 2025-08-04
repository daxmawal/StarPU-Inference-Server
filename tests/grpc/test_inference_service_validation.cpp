#include <gtest/gtest.h>

#include <thread>

#include "grpc/server/inference_service.hpp"

using namespace starpu_server;

static inference::ModelInferRequest
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

TEST(InferenceServiceValidation, RawInputCountMismatch)
{
  auto req = make_valid_request();
  // add an extra raw input to cause mismatch
  req.add_raw_input_contents()->assign("", 0);
  std::vector<torch::Tensor> inputs;
  auto status = InferenceServiceImpl::validate_and_convert_inputs(&req, inputs);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceServiceValidation, RawContentSizeMismatch)
{
  inference::ModelInferRequest req;
  auto* input = req.add_inputs();
  input->set_name("input0");
  input->set_datatype("FP32");
  input->add_shape(2);
  input->add_shape(2);
  // only 3 floats provided but shape expects 4
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  req.add_raw_input_contents()->assign(
      reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
  std::vector<torch::Tensor> inputs;
  auto status = InferenceServiceImpl::validate_and_convert_inputs(&req, inputs);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceServiceValidation, SubmitJobAndWaitInternalError)
{
  InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs = {torch::zeros({1})};
  InferenceServiceImpl service(&queue, &ref_outputs);

  std::vector<torch::Tensor> inputs = {torch::tensor({1})};
  std::vector<torch::Tensor> outputs;

  std::thread worker([&] {
    std::shared_ptr<InferenceJob> job;
    queue.wait_and_pop(job);
    job->get_on_complete()({}, 0.0);
  });

  auto status = service.submit_job_and_wait(inputs, outputs);
  worker.join();

  EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
}

TEST(InferenceServiceValidation, PopulateResponseSetsServerTimes)
{
  inference::ModelInferRequest req;
  req.set_model_name("model");
  req.set_model_version("1");

  std::vector<torch::Tensor> outputs = {
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kInt))};
  inference::ModelInferResponse reply;
  int64_t recv_ms = 10;
  int64_t send_ms = 20;

  InferenceServiceImpl::populate_response(
      &req, &reply, outputs, recv_ms, send_ms);

  EXPECT_EQ(reply.server_receive_ms(), recv_ms);
  EXPECT_EQ(reply.server_send_ms(), send_ms);
}
