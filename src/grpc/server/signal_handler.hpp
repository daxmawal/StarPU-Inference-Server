#pragma once

#include <grpcpp/server.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>

namespace starpu_server {

class InferenceQueue;

struct ServerContext {
  InferenceQueue* queue_ptr = nullptr;
  std::unique_ptr<grpc::Server> server;
  std::mutex stop_mutex;
  std::condition_variable stop_cv;
  std::atomic<bool> stop_requested{false};
};

auto server_context() -> ServerContext&;

void signal_handler(int signal);

}  // namespace starpu_server
