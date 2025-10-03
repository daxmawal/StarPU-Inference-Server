#include <grpcpp/grpcpp.h>
#include <torch/script.h>

#include <chrono>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "client_args.hpp"
#include "grpc_service.grpc.pb.h"
#include "inference_client.hpp"
#include "utils/input_generator.hpp"
#include "utils/logger.hpp"

auto
main(int argc, char* argv[]) -> int
{
  std::vector<const char*> const_argv(argv, argv + argc);
  std::span<const char*> args{const_argv};
  const starpu_server::ClientConfig config =
      starpu_server::parse_client_args(args);
  if (config.show_help) {
    starpu_server::display_client_help(args.front());
    return 0;
  }
  if (!config.valid) {
    starpu_server::log_error("Invalid program options.");
    return 1;
  }
  grpc::ChannelArguments ch_args;
  const int max_msg_size = 32 * 1024 * 1024;
  ch_args.SetMaxReceiveMessageSize(max_msg_size);
  ch_args.SetMaxSendMessageSize(max_msg_size);

  auto channel = grpc::CreateCustomChannel(
      config.server_address, grpc::InsecureChannelCredentials(), ch_args);

  starpu_server::InferenceClient client(channel, config.verbosity);

  if (!client.ServerIsLive()) {
    return 1;
  }

  if (!client.ServerIsReady()) {
    return 1;
  }

  if (!client.ModelIsReady({config.model_name, config.model_version})) {
    return 1;
  }

  constexpr int NUM_TENSORS = 5;
  std::vector<std::vector<torch::Tensor>> tensor_pool;
  tensor_pool.reserve(NUM_TENSORS);
  for (int i = 0; i < NUM_TENSORS; ++i) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(config.inputs.size());
    for (size_t j = 0; j < config.inputs.size(); ++j) {
      const auto& in_cfg = config.inputs[j];
      tensors.push_back(starpu_server::input_generator::generate_random_tensor(
          in_cfg.shape, in_cfg.type, j));
    }
    tensor_pool.push_back(std::move(tensors));
  }

  thread_local std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution dist(0, NUM_TENSORS - 1);

  std::jthread cq_thread(
      &starpu_server::InferenceClient::AsyncCompleteRpc, &client);

  auto next_time = std::chrono::steady_clock::now();
  const auto delay = std::chrono::milliseconds(config.delay_ms);
  for (int i = 0; i < config.iterations; ++i) {
    std::this_thread::sleep_until(next_time);
    next_time += delay;

    const auto idx = static_cast<size_t>(dist(rng));
    client.AsyncModelInfer(tensor_pool[idx], config);
  }

  client.Shutdown();

  return 0;
}
