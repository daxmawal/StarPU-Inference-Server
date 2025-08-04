#include <gtest/gtest.h>

#include <cstring>

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

TEST(InferenceService, ValidateInputsSuccess)
{
  auto req = make_valid_request();
  std::vector<torch::Tensor> inputs;
  auto status = InferenceServiceImpl::validate_and_convert_inputs(&req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1u);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(inputs[0].scalar_type(), at::kFloat);
  EXPECT_FLOAT_EQ(inputs[0][0][0].item<float>(), 1.0f);
}

TEST(InferenceService, ValidateInputsMultipleDtypes)
{
  inference::ModelInferRequest req;

  auto* input0 = req.add_inputs();
  input0->set_name("input0");
  input0->set_datatype("FP32");
  input0->add_shape(2);
  input0->add_shape(2);
  std::vector<float> data0 = {1.0f, 2.0f, 3.0f, 4.0f};
  req.add_raw_input_contents()->assign(
      reinterpret_cast<const char*>(data0.data()),
      data0.size() * sizeof(float));

  auto* input1 = req.add_inputs();
  input1->set_name("input1");
  input1->set_datatype("INT64");
  input1->add_shape(3);
  std::vector<int64_t> data1 = {10, 20, 30};
  req.add_raw_input_contents()->assign(
      reinterpret_cast<const char*>(data1.data()),
      data1.size() * sizeof(int64_t));

  std::vector<torch::Tensor> inputs;
  auto status = InferenceServiceImpl::validate_and_convert_inputs(&req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 2u);

  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(inputs[0].scalar_type(), at::kFloat);
  EXPECT_FLOAT_EQ(inputs[0][0][0].item<float>(), 1.0f);

  EXPECT_EQ(inputs[1].sizes(), (torch::IntArrayRef{3}));
  EXPECT_EQ(inputs[1].scalar_type(), at::kLong);
  EXPECT_EQ(inputs[1][0].item<int64_t>(), 10);
}

TEST(InferenceService, RawInputCountMismatch)
{
  auto req = make_valid_request();
  req.add_raw_input_contents()->assign("", 0);
  std::vector<torch::Tensor> inputs;
  auto status = InferenceServiceImpl::validate_and_convert_inputs(&req, inputs);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, RawContentSizeMismatch)
{
  inference::ModelInferRequest req;
  auto* input = req.add_inputs();
  input->set_name("input0");
  input->set_datatype("FP32");
  input->add_shape(2);
  input->add_shape(2);
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  req.add_raw_input_contents()->assign(
      reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
  std::vector<torch::Tensor> inputs;
  auto status = InferenceServiceImpl::validate_and_convert_inputs(&req, inputs);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, PopulateResponseFillsFields)
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

  EXPECT_EQ(reply.model_name(), "model");
  EXPECT_EQ(reply.model_version(), "1");
  EXPECT_EQ(reply.server_receive_ms(), recv_ms);
  EXPECT_EQ(reply.server_send_ms(), send_ms);

  ASSERT_EQ(reply.outputs_size(), 1);
  ASSERT_EQ(reply.raw_output_contents_size(), 1);

  const auto& out = reply.outputs(0);
  EXPECT_EQ(out.name(), "output0");
  EXPECT_EQ(out.datatype(), "INT32");
  ASSERT_EQ(out.shape_size(), 1);
  EXPECT_EQ(out.shape(0), 3);

  auto flat = outputs[0].view({-1});
  const auto& raw = reply.raw_output_contents(0);
  ASSERT_EQ(raw.size(), flat.numel() * flat.element_size());
  EXPECT_EQ(0, std::memcmp(raw.data(), flat.data_ptr(), raw.size()));
}

TEST(InferenceService, SubmitJobAndWaitReturnsInternalOnEmptyOutput)
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

TEST(InferenceService, ModelInferReturnsValidationError)
{
  InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  InferenceServiceImpl service(&queue, &ref_outputs);

  auto req = make_valid_request();
  req.add_raw_input_contents()->assign("", 0);
  req.set_model_name("m");
  req.set_model_version("1");

  grpc::ServerContext ctx;
  inference::ModelInferResponse reply;

  auto status = service.ModelInfer(&ctx, &req, &reply);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(reply.model_name(), "");
  EXPECT_EQ(reply.model_version(), "");
  EXPECT_EQ(reply.outputs_size(), 0);
  EXPECT_EQ(reply.raw_output_contents_size(), 0);
  EXPECT_EQ(reply.server_receive_ms(), 0);
  EXPECT_EQ(reply.server_send_ms(), 0);
}

TEST(InferenceService, ModelInferPropagatesSubmitError)
{
  InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs = {torch::zeros({1})};
  InferenceServiceImpl service(&queue, &ref_outputs);

  auto req = make_valid_request();
  req.set_model_name("m");
  req.set_model_version("1");

  std::thread worker([&] {
    std::shared_ptr<InferenceJob> job;
    queue.wait_and_pop(job);
    job->get_on_complete()({}, 0.0);
  });

  grpc::ServerContext ctx;
  inference::ModelInferResponse reply;
  auto status = service.ModelInfer(&ctx, &req, &reply);
  worker.join();

  EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
  EXPECT_EQ(reply.model_name(), "");
  EXPECT_EQ(reply.model_version(), "");
  EXPECT_EQ(reply.outputs_size(), 0);
  EXPECT_EQ(reply.raw_output_contents_size(), 0);
  EXPECT_EQ(reply.server_receive_ms(), 0);
  EXPECT_EQ(reply.server_send_ms(), 0);
}

TEST(InferenceService, ModelInferReturnsOutputs)
{
  InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs = {torch::zeros({2, 2})};
  InferenceServiceImpl service(&queue, &ref_outputs);

  auto req = make_valid_request();
  req.set_model_name("m");
  req.set_model_version("1");

  std::thread worker([&] {
    std::shared_ptr<InferenceJob> job;
    queue.wait_and_pop(job);
    std::vector<torch::Tensor> outs = {
        torch::tensor({10.0f, 20.0f, 30.0f, 40.0f}).view({2, 2})};
    job->get_on_complete()(outs, 0.0);
  });

  grpc::ServerContext ctx;
  inference::ModelInferResponse reply;
  auto status = service.ModelInfer(&ctx, &req, &reply);
  worker.join();

  ASSERT_TRUE(status.ok());
  EXPECT_EQ(reply.model_name(), "m");
  EXPECT_EQ(reply.model_version(), "1");
  EXPECT_GT(reply.server_receive_ms(), 0);
  EXPECT_GT(reply.server_send_ms(), 0);

  ASSERT_EQ(reply.outputs_size(), 1);
  ASSERT_EQ(reply.raw_output_contents_size(), 1);

  const auto& out = reply.outputs(0);
  EXPECT_EQ(out.datatype(), "FP32");
  ASSERT_EQ(out.shape_size(), 2);
  EXPECT_EQ(out.shape(0), 2);
  EXPECT_EQ(out.shape(1), 2);

  const auto& raw = reply.raw_output_contents(0);
  std::vector<float> expected = {10.0f, 20.0f, 30.0f, 40.0f};
  ASSERT_EQ(raw.size(), expected.size() * sizeof(float));
  EXPECT_EQ(0, std::memcmp(raw.data(), expected.data(), raw.size()));
}

TEST(GrpcServer, StartAndStop)
{
  InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  std::unique_ptr<grpc::Server> server;

  std::thread server_thread([&]() {
    RunGrpcServer(queue, reference_outputs, "127.0.0.1:0", 4, server);
  });

  while (!server) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  StopServer(server);
  server_thread.join();

  EXPECT_EQ(server, nullptr);
}

TEST(GrpcServer, StopServerNullptr)
{
  std::unique_ptr<grpc::Server> server;
  EXPECT_NO_THROW(StopServer(server));
  EXPECT_EQ(server, nullptr);
}
