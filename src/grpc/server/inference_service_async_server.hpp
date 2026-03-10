inline namespace inference_service_detail {

template <typename Request, typename Response>
class UnaryCallData final
    : public AsyncCallDataBase,
      public std::enable_shared_from_this<UnaryCallData<Request, Response>> {
 public:
  using Self = UnaryCallData<Request, Response>;
  using SharedPtr = std::shared_ptr<Self>;
  using RequestMethod = void (inference::GRPCInferenceService::AsyncService::*)(
      grpc::ServerContext*, Request*,
      grpc::ServerAsyncResponseWriter<Response>*, grpc::CompletionQueue*,
      grpc::ServerCompletionQueue*, void*);
  using Handler = std::function<grpc::Status(
      InferenceServiceImpl*, grpc::ServerContext*, const Request*, Response*)>;

  UnaryCallData(
      inference::GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* completion_queue, InferenceServiceImpl* impl,
      RequestMethod request_method, Handler handler)
      : service_(service), cq_(completion_queue), responder_(&ctx_),
        impl_(impl), request_method_(request_method),
        handler_(std::move(handler))
  {
  }

  static void Start(
      inference::GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* completion_queue, InferenceServiceImpl* impl,
      RequestMethod request_method, Handler handler)
  {
    auto call = std::make_shared<Self>(
        service, completion_queue, impl, request_method, std::move(handler));
    call->Proceed(true);
  }

  void Proceed(bool is_ok) override
  {
    switch (status_) {
      case CallStatus::Create: {
        status_ = CallStatus::Process;
        self_ref_ = this->shared_from_this();
        (service_->*request_method_)(
            &ctx_, &request_, &responder_, cq_, cq_, this);
        break;
      }
      case CallStatus::Process: {
        if (!is_ok) {
          status_ = CallStatus::Finish;
          self_ref_.reset();
          return;
        }
        Start(service_, cq_, impl_, request_method_, handler_);
        HandleRequest();
        break;
      }
      case CallStatus::Finish:
        self_ref_.reset();
        break;
    }
  }

 private:
  enum class CallStatus : std::uint8_t { Create, Process, Finish };

  void HandleRequest()
  {
    if (!handler_) {
      status_ = CallStatus::Finish;
      responder_.Finish(
          response_,
          {grpc::StatusCode::INTERNAL, "Unary handler not configured"}, this);
      return;
    }
    auto status = handler_(impl_, &ctx_, &request_, &response_);
    status_ = CallStatus::Finish;
    responder_.Finish(response_, status, this);
  }

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
 public:
  void HandleRequestWithoutResponderForTest()
  {
    if (!handler_) {
      status_ = CallStatus::Finish;
      return;
    }
    HandleRequest();
  }

  [[nodiscard]] auto StatusIsFinishForTest() const -> bool
  {
    return status_ == CallStatus::Finish;
  }
#endif  // SONAR_IGNORE_END

  inference::GRPCInferenceService::AsyncService* service_;
  grpc::ServerCompletionQueue* cq_;
  grpc::ServerContext ctx_;
  Request request_;
  Response response_;
  grpc::ServerAsyncResponseWriter<Response> responder_;
  InferenceServiceImpl* impl_;
  RequestMethod request_method_;
  Handler handler_;
  CallStatus status_ = CallStatus::Create;
  SharedPtr self_ref_;
};

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto
unary_call_data_missing_handler_transitions_to_finish_for_test_impl() -> bool
{
  using ServerLiveUnaryCallData = UnaryCallData<
      inference::ServerLiveRequest, inference::ServerLiveResponse>;
  ServerLiveUnaryCallData call_data(
      /*service=*/nullptr,
      /*completion_queue=*/nullptr,
      /*impl=*/nullptr,
      /*request_method=*/ServerLiveUnaryCallData::RequestMethod{},
      /*handler=*/ServerLiveUnaryCallData::Handler{});
  call_data.HandleRequestWithoutResponderForTest();
  return call_data.StatusIsFinishForTest();
}
#endif  // SONAR_IGNORE_END

class ModelInferCallData final
    : public AsyncCallDataBase,
      public std::enable_shared_from_this<ModelInferCallData> {
 public:
  using Self = ModelInferCallData;
  using SharedPtr = std::shared_ptr<Self>;

  ModelInferCallData(
      inference::GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* completion_queue, InferenceServiceImpl* impl)
      : service_(service), cq_(completion_queue), responder_(&ctx_), impl_(impl)
  {
  }

  static void Start(
      inference::GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* completion_queue, InferenceServiceImpl* impl)
  {
    auto call = std::make_shared<Self>(service, completion_queue, impl);
    call->Proceed(true);
  }

  void Proceed(bool is_ok) override
  {
    using enum CallStatus;
    switch (status_) {
      case Create: {
        status_ = Process;
        self_ref_ = this->shared_from_this();
        service_->RequestModelInfer(
            &ctx_, &request_, &responder_, cq_, cq_, this);
        break;
      }
      case Process: {
        if (!is_ok) {
          status_ = Finish;
          self_ref_.reset();
          return;
        }
        Start(service_, cq_, impl_);
        auto self = this->shared_from_this();
        auto call_guard = self;
        impl_->HandleModelInferAsync(
            &ctx_, &request_, &response_,
            [self = std::move(self)](const Status& status) {
              self->OnInferenceComplete(status);
            },
            std::move(call_guard));
        break;
      }
      case Finish:
        self_ref_.reset();
        break;
    }
  }

 private:
  enum class CallStatus : std::uint8_t { Create, Process, Finish };

  void OnInferenceComplete(const Status& status)
  {
    status_ = CallStatus::Finish;
    responder_.Finish(response_, status, this);
  }

  inference::GRPCInferenceService::AsyncService* service_;
  grpc::ServerCompletionQueue* cq_;
  grpc::ServerContext ctx_;
  ModelInferRequest request_;
  ModelInferResponse response_;
  grpc::ServerAsyncResponseWriter<ModelInferResponse> responder_;
  InferenceServiceImpl* impl_;
  CallStatus status_ = CallStatus::Create;
  SharedPtr self_ref_;
};

class ModelStreamInferCallData final
    : public AsyncCallDataBase,
      public std::enable_shared_from_this<ModelStreamInferCallData> {
 public:
  using Self = ModelStreamInferCallData;
  using SharedPtr = std::shared_ptr<Self>;

  ModelStreamInferCallData(
      inference::GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* completion_queue)
      : service_(service), cq_(completion_queue), stream_(&ctx_)
  {
  }

  static void Start(
      inference::GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* completion_queue)
  {
    auto call = std::make_shared<Self>(service, completion_queue);
    call->Proceed(true);
  }

  void Proceed(bool is_ok) override
  {
    using enum CallStatus;
    switch (status_) {
      case Create: {
        status_ = Process;
        self_ref_ = this->shared_from_this();
        service_->RequestModelStreamInfer(&ctx_, &stream_, cq_, cq_, this);
        break;
      }
      case Process: {
        if (!is_ok) {
          status_ = Finish;
          self_ref_.reset();
          return;
        }
        Start(service_, cq_);
        status_ = Finish;
        stream_.Finish(unimplemented_rpc_status("ModelStreamInfer"), this);
        break;
      }
      case Finish:
        self_ref_.reset();
        break;
    }
  }

 private:
  enum class CallStatus : std::uint8_t { Create, Process, Finish };

  inference::GRPCInferenceService::AsyncService* service_;
  grpc::ServerCompletionQueue* cq_;
  grpc::ServerContext ctx_;
  grpc::ServerAsyncReaderWriter<
      inference::ModelStreamInferResponse, inference::ModelInferRequest>
      stream_;
  CallStatus status_ = CallStatus::Create;
  SharedPtr self_ref_;
};

auto
compute_thread_count() -> std::size_t
{
  return compute_thread_count_from(std::thread::hardware_concurrency());
}

using UnaryRpcRegistrationFn = void (*)(
    inference::GRPCInferenceService::AsyncService*,
    grpc::ServerCompletionQueue*, InferenceServiceImpl*);

struct UnaryRpcDescriptor {
  UnaryRpcRegistrationFn register_fn = nullptr;
};

template <
    typename Request, typename Response, auto kRequestMethod,
    auto kHandlerMethod>
void
start_unary_rpc_entry(
    inference::GRPCInferenceService::AsyncService* service,
    grpc::ServerCompletionQueue* completion_queue, InferenceServiceImpl* impl)
{
  UnaryCallData<Request, Response>::Start(
      service, completion_queue, impl,
      static_cast<typename UnaryCallData<Request, Response>::RequestMethod>(
          kRequestMethod),
      std::mem_fn(kHandlerMethod));
}

template <
    typename Request, typename Response, auto kRequestMethod,
    auto kHandlerMethod>
constexpr auto
make_unary_rpc_descriptor() -> UnaryRpcDescriptor
{
  return {&start_unary_rpc_entry<
      Request, Response, kRequestMethod, kHandlerMethod>};
}

auto
unary_rpc_descriptors() -> std::span<const UnaryRpcDescriptor>
{
  static constexpr std::array kUnaryRpcDescriptors = {
      make_unary_rpc_descriptor<
          inference::ServerLiveRequest, inference::ServerLiveResponse,
          &inference::GRPCInferenceService::AsyncService::RequestServerLive,
          &InferenceServiceImpl::ServerLive>(),
      make_unary_rpc_descriptor<
          inference::ServerReadyRequest, inference::ServerReadyResponse,
          &inference::GRPCInferenceService::AsyncService::RequestServerReady,
          &InferenceServiceImpl::ServerReady>(),
      make_unary_rpc_descriptor<
          inference::ModelReadyRequest, inference::ModelReadyResponse,
          &inference::GRPCInferenceService::AsyncService::RequestModelReady,
          &InferenceServiceImpl::ModelReady>(),
      make_unary_rpc_descriptor<
          inference::ServerMetadataRequest, inference::ServerMetadataResponse,
          &inference::GRPCInferenceService::AsyncService::RequestServerMetadata,
          &InferenceServiceImpl::ServerMetadata>(),
      make_unary_rpc_descriptor<
          inference::ModelMetadataRequest, inference::ModelMetadataResponse,
          &inference::GRPCInferenceService::AsyncService::RequestModelMetadata,
          &InferenceServiceImpl::ModelMetadata>(),
      make_unary_rpc_descriptor<
          inference::ModelConfigRequest, inference::ModelConfigResponse,
          &inference::GRPCInferenceService::AsyncService::RequestModelConfig,
          &InferenceServiceImpl::ModelConfig>(),
      make_unary_rpc_descriptor<
          inference::ModelStatisticsRequest, inference::ModelStatisticsResponse,
          &inference::GRPCInferenceService::AsyncService::
              RequestModelStatistics,
          &InferenceServiceImpl::ModelStatistics>(),
      make_unary_rpc_descriptor<
          inference::RepositoryIndexRequest, inference::RepositoryIndexResponse,
          &inference::GRPCInferenceService::AsyncService::
              RequestRepositoryIndex,
          &InferenceServiceImpl::RepositoryIndex>(),
      make_unary_rpc_descriptor<
          inference::RepositoryModelLoadRequest,
          inference::RepositoryModelLoadResponse,
          &inference::GRPCInferenceService::AsyncService::
              RequestRepositoryModelLoad,
          &InferenceServiceImpl::RepositoryModelLoad>(),
      make_unary_rpc_descriptor<
          inference::RepositoryModelUnloadRequest,
          inference::RepositoryModelUnloadResponse,
          &inference::GRPCInferenceService::AsyncService::
              RequestRepositoryModelUnload,
          &InferenceServiceImpl::RepositoryModelUnload>(),
      make_unary_rpc_descriptor<
          inference::SystemSharedMemoryStatusRequest,
          inference::SystemSharedMemoryStatusResponse,
          &inference::GRPCInferenceService::AsyncService::
              RequestSystemSharedMemoryStatus,
          &InferenceServiceImpl::SystemSharedMemoryStatus>(),
      make_unary_rpc_descriptor<
          inference::SystemSharedMemoryRegisterRequest,
          inference::SystemSharedMemoryRegisterResponse,
          &inference::GRPCInferenceService::AsyncService::
              RequestSystemSharedMemoryRegister,
          &InferenceServiceImpl::SystemSharedMemoryRegister>(),
      make_unary_rpc_descriptor<
          inference::SystemSharedMemoryUnregisterRequest,
          inference::SystemSharedMemoryUnregisterResponse,
          &inference::GRPCInferenceService::AsyncService::
              RequestSystemSharedMemoryUnregister,
          &InferenceServiceImpl::SystemSharedMemoryUnregister>(),
      make_unary_rpc_descriptor<
          inference::CudaSharedMemoryStatusRequest,
          inference::CudaSharedMemoryStatusResponse,
          &inference::GRPCInferenceService::AsyncService::
              RequestCudaSharedMemoryStatus,
          &InferenceServiceImpl::CudaSharedMemoryStatus>(),
      make_unary_rpc_descriptor<
          inference::CudaSharedMemoryRegisterRequest,
          inference::CudaSharedMemoryRegisterResponse,
          &inference::GRPCInferenceService::AsyncService::
              RequestCudaSharedMemoryRegister,
          &InferenceServiceImpl::CudaSharedMemoryRegister>(),
      make_unary_rpc_descriptor<
          inference::CudaSharedMemoryUnregisterRequest,
          inference::CudaSharedMemoryUnregisterResponse,
          &inference::GRPCInferenceService::AsyncService::
              RequestCudaSharedMemoryUnregister,
          &InferenceServiceImpl::CudaSharedMemoryUnregister>(),
      make_unary_rpc_descriptor<
          inference::TraceSettingRequest, inference::TraceSettingResponse,
          &inference::GRPCInferenceService::AsyncService::RequestTraceSetting,
          &InferenceServiceImpl::TraceSetting>(),
      make_unary_rpc_descriptor<
          inference::LogSettingsRequest, inference::LogSettingsResponse,
          &inference::GRPCInferenceService::AsyncService::RequestLogSettings,
          &InferenceServiceImpl::LogSettings>(),
  };
  return kUnaryRpcDescriptors;
}

void
register_async_unary_rpcs(
    inference::GRPCInferenceService::AsyncService* service,
    grpc::ServerCompletionQueue* completion_queue, InferenceServiceImpl* impl)
{
  for (const auto& descriptor : unary_rpc_descriptors()) {
    descriptor.register_fn(service, completion_queue, impl);
  }
}

inline namespace inference_service_detail {

void
enable_grpc_health_and_reflection()
{
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    grpc::EnableDefaultHealthCheckService(true);
#if defined(STARPU_ENABLE_GRPC_REFLECTION)
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
#endif
  });
}

}  // namespace inference_service_detail

void
run_grpc_server_impl(
    InferenceServiceImpl& service, const GrpcServerOptions& options,
    std::unique_ptr<Server>& server, const GrpcServerLifecycleHooks& hooks)
{
  inference::GRPCInferenceService::AsyncService async_service;
  AsyncServerContext async_context(async_service, service);

  enable_grpc_health_and_reflection();

  ServerBuilder builder;
  builder.AddListeningPort(options.address, grpc::InsecureServerCredentials());
  async_context.configure(builder);
  const int grpc_max_message_bytes =
      options.max_message_bytes >
              static_cast<std::size_t>(std::numeric_limits<int>::max())
          ? std::numeric_limits<int>::max()
          : static_cast<int>(options.max_message_bytes);
  builder.SetMaxReceiveMessageSize(grpc_max_message_bytes);
  builder.SetMaxSendMessageSize(grpc_max_message_bytes);

  server = builder.BuildAndStart();
  if (!server) {
    log_error(
        std::format("Failed to start gRPC server on {}", options.address));
    set_server_health(false);
    if (hooks.on_stopped) {
      hooks.on_stopped();
    }
    return;
  }
  set_server_health(true);
  set_grpc_health_status(server.get(), true);
  if (hooks.on_started) {
    hooks.on_started(server.get());
  }
  async_context.start();
  log_info(
      options.verbosity,
      std::format("Server listening on {}", options.address));
  server->Wait();
  set_server_health(false);
  set_grpc_health_status(server.get(), false);
  async_context.shutdown();
  if (hooks.on_stopped) {
    hooks.on_stopped();
  }
  server.reset();
}

}  // namespace inference_service_detail

AsyncServerContext::AsyncServerContext(
    inference::GRPCInferenceService::AsyncService& async_service,
    InferenceServiceImpl& impl)
    : async_service_(&async_service), impl_(&impl)
{
}

void
AsyncServerContext::configure(grpc::ServerBuilder& builder)
{
  builder.RegisterService(async_service_);
  completion_queue_ = builder.AddCompletionQueue();
}

void
AsyncServerContext::start()
{
  if (!completion_queue_ || started_) {
    return;
  }
  started_ = true;
  const std::size_t thread_count = compute_thread_count();
  threads_.reserve(thread_count);
  for (std::size_t i = 0; i < thread_count; ++i) {
    threads_.emplace_back([this]() { this->poll_events(); });
  }

  register_async_unary_rpcs(async_service_, completion_queue_.get(), impl_);
  ModelStreamInferCallData::Start(async_service_, completion_queue_.get());
  ModelInferCallData::Start(async_service_, completion_queue_.get(), impl_);
}

void
AsyncServerContext::shutdown()
{
  if (!started_) {
    return;
  }
  started_ = false;
  if (completion_queue_) {
    completion_queue_->Shutdown();
  }
  threads_.clear();
  completion_queue_.reset();
}

void
AsyncServerContext::poll_events()
{
  void* tag = nullptr;
  bool event_ok = false;
  while (completion_queue_ && completion_queue_->Next(&tag, &event_ok)) {
    static_cast<AsyncCallDataBase*>(tag)->Proceed(event_ok);
  }
}

void
RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const GrpcModelSpec& model_spec, const GrpcServerOptions& options,
    std::unique_ptr<Server>& server, const GrpcServerLifecycleHooks& hooks)
{
  auto service_options = InferenceServiceImpl::ServiceOptions{
      .default_model_name = options.default_model_name,
      .expected_input_names = std::vector<std::string>(
          model_spec.expected_input_names.begin(),
          model_spec.expected_input_names.end()),
      .expected_output_names = std::vector<std::string>(
          model_spec.expected_output_names.begin(),
          model_spec.expected_output_names.end()),
      .server_name = options.server_name,
      .server_version = options.server_version};
  InferenceServiceImpl service(
      &queue, &reference_outputs,
      std::vector<at::ScalarType>(
          model_spec.expected_input_types.begin(),
          model_spec.expected_input_types.end()),
      InferenceServiceImpl::InputShapeConfig{
          std::vector<std::vector<int64_t>>(
              model_spec.expected_input_dims.begin(),
              model_spec.expected_input_dims.end()),
          model_spec.max_batch_size},
      std::move(service_options));
  run_grpc_server_impl(service, options, server, hooks);
}

void
RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::vector<at::ScalarType>& expected_input_types,
    GrpcModelNamesSpec model_names, const GrpcServerOptions& options,
    std::unique_ptr<Server>& server, const GrpcServerLifecycleHooks& hooks)
{
  const GrpcModelSpec model_spec{
      .expected_input_types = expected_input_types,
      .expected_input_dims = {},
      .expected_input_names = model_names.expected_input_names,
      .expected_output_names = model_names.expected_output_names,
      .max_batch_size = 0};
  RunGrpcServer(queue, reference_outputs, model_spec, options, server, hooks);
}

void
StopServer(Server* server)
{
  if (server != nullptr) {
    set_grpc_health_status(server, false);
    server->Shutdown();
  }
}
