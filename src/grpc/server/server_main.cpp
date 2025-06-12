#include <ATen/ATen.h>
#include <grpcpp/grpcpp.h>
#include <torch/script.h>

#include <iostream>
#include <memory>
#include <string>

#include "cli/args_parser.hpp"
#include "core/inference_runner.hpp"
#include "core/starpu_setup.hpp"
#include "core/warmup.hpp"
#include "grpc_service.grpc.pb.h"
#include "server/Inference_queue.hpp"
#include "server/server_worker.hpp"
#include "utils/client_utils.hpp"

// Forward declarations from inference_runner.cpp
auto load_model_and_reference_output(const RuntimeConfig& opts) -> std::tuple<
    torch::jit::script::Module, std::vector<torch::jit::script::Module>,
    std::vector<torch::Tensor>>;
void run_warmup(
    const RuntimeConfig& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<torch::Tensor>& outputs_ref);

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using inference::GRPCInferenceService;
using inference::ModelInferRequest;
using inference::ModelInferResponse;
using inference::ServerLiveRequest;
using inference::ServerLiveResponse;

class InferenceServiceImpl final : public GRPCInferenceService::Service {
 public:
  InferenceServiceImpl(
      InferenceQueue& queue, const std::vector<torch::Tensor>& outputs_ref)
      : queue_(queue), outputs_ref_(outputs_ref)
  {
  }

  Status ServerLive(
      ServerContext* /*context*/, const ServerLiveRequest* /*request*/,
      ServerLiveResponse* reply) override
  {
    reply->set_live(true);
    return Status::OK;
  }

  Status ModelInfer(
      ServerContext* /*context*/, const ModelInferRequest* request,
      ModelInferResponse* reply) override
  {
    std::vector<torch::Tensor> inputs;
    inputs.reserve(static_cast<size_t>(request->inputs_size()));

    for (const auto& input : request->inputs()) {
      std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

      auto tensor = torch::empty(shape, torch::kFloat32);
      float* dest = tensor.data_ptr<float>();
      const auto& contents = input.contents().fp32_contents();
      for (int i = 0; i < contents.size(); ++i) {
        dest[i] = contents.Get(i);
      }
      inputs.push_back(tensor);
    }

    auto job = client_utils::create_job(inputs, outputs_ref_, next_job_id_++);

    std::promise<std::vector<torch::Tensor>> prom;
    auto fut = prom.get_future();
    job->set_on_complete(
        [&prom](const std::vector<torch::Tensor>& outs, double) {
          prom.set_value(outs);
        });

    queue_.push(job);

    auto outputs = fut.get();

    reply->set_model_name(request->model_name());
    reply->set_model_version(request->model_version());

    for (size_t idx = 0; idx < outputs.size(); ++idx) {
      const auto& out = outputs[idx].to(torch::kCPU);
      auto* out_tensor = reply->add_outputs();
      out_tensor->set_name("output" + std::to_string(idx));
      out_tensor->set_datatype("FP32");
      for (const auto dim : out.sizes()) {
        out_tensor->add_shape(dim);
      }

      auto flat = out.view({-1});
      auto* contents = out_tensor->mutable_contents();
      contents->mutable_fp32_contents()->Reserve(flat.numel());
      for (int64_t i = 0; i < flat.numel(); ++i) {
        contents->add_fp32_contents(flat[i].item<float>());
      }
    }

    return Status::OK;
  }

 private:
  InferenceQueue& queue_;
  const std::vector<torch::Tensor>& outputs_ref_;
  std::atomic<unsigned int> next_job_id_{0};
};

void
RunServer(InferenceQueue& queue, const std::vector<torch::Tensor>& outputs_ref)
{
  std::string server_address("0.0.0.0:50051");
  InferenceServiceImpl service(queue, outputs_ref);

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  const int max_msg_size = 32 * 1024 * 1024;  // 32MB
  builder.SetMaxReceiveMessageSize(max_msg_size);
  builder.SetMaxSendMessageSize(max_msg_size);

  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listen on " << server_address << std::endl;
  server->Wait();
}

int
main(int argc, char* argv[])
{
  RuntimeConfig opts =
      parse_arguments(std::span<char*>(argv, static_cast<size_t>(argc)));
  if (opts.show_help) {
    display_help("grpc_server");
    return 0;
  }
  if (!opts.valid) {
    log_fatal("Invalid program options.\n");
  }

  StarPUSetup starpu(opts);

  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> outputs_ref;
  std::tie(model_cpu, models_gpu, outputs_ref) =
      load_model_and_reference_output(opts);

  run_warmup(opts, starpu, model_cpu, models_gpu, outputs_ref);

  InferenceQueue queue;
  std::vector<InferenceResult> results;
  std::mutex results_mutex;
  std::atomic<unsigned int> completed_jobs = 0;
  std::condition_variable all_done_cv;

  ServerWorker worker(
      &queue, &model_cpu, &models_gpu, &starpu, &opts, &results, &results_mutex,
      &completed_jobs, &all_done_cv);

  std::thread worker_thread(&ServerWorker::run, &worker);

  RunServer(queue, outputs_ref);

  queue.shutdown();
  worker_thread.join();

  cudaDeviceSynchronize();

  return 0;
}