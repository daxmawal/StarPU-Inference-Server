namespace {

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

namespace {

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

}  // namespace

void
run_grpc_server_impl(
    InferenceServiceImpl& service, const GrpcServerOptions& options,
    std::unique_ptr<Server>& server)
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
    return;
  }
  set_server_health(true);
  set_grpc_health_status(server.get(), true);
  async_context.start();
  log_info(
      options.verbosity,
      std::format("Server listening on {}", options.address));
  server->Wait();
  set_server_health(false);
  set_grpc_health_status(server.get(), false);
  async_context.shutdown();
  server.reset();
}

}  // namespace

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

  UnaryCallData<inference::ServerLiveRequest, inference::ServerLiveResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::RequestServerLive,
          std::mem_fn(&InferenceServiceImpl::ServerLive));
  UnaryCallData<inference::ServerReadyRequest, inference::ServerReadyResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::RequestServerReady,
          std::mem_fn(&InferenceServiceImpl::ServerReady));
  UnaryCallData<inference::ModelReadyRequest, inference::ModelReadyResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::RequestModelReady,
          std::mem_fn(&InferenceServiceImpl::ModelReady));
  UnaryCallData<
      inference::ServerMetadataRequest, inference::ServerMetadataResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::RequestServerMetadata,
          std::mem_fn(&InferenceServiceImpl::ServerMetadata));
  UnaryCallData<
      inference::ModelMetadataRequest, inference::ModelMetadataResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::RequestModelMetadata,
          std::mem_fn(&InferenceServiceImpl::ModelMetadata));
  UnaryCallData<inference::ModelConfigRequest, inference::ModelConfigResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::RequestModelConfig,
          std::mem_fn(&InferenceServiceImpl::ModelConfig));
  UnaryCallData<
      inference::ModelStatisticsRequest, inference::ModelStatisticsResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::
              RequestModelStatistics,
          std::mem_fn(&InferenceServiceImpl::ModelStatistics));
  UnaryCallData<
      inference::RepositoryIndexRequest, inference::RepositoryIndexResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::
              RequestRepositoryIndex,
          std::mem_fn(&InferenceServiceImpl::RepositoryIndex));
  UnaryCallData<
      inference::RepositoryModelLoadRequest,
      inference::RepositoryModelLoadResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::
              RequestRepositoryModelLoad,
          std::mem_fn(&InferenceServiceImpl::RepositoryModelLoad));
  UnaryCallData<
      inference::RepositoryModelUnloadRequest,
      inference::RepositoryModelUnloadResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::
              RequestRepositoryModelUnload,
          std::mem_fn(&InferenceServiceImpl::RepositoryModelUnload));
  UnaryCallData<
      inference::SystemSharedMemoryStatusRequest,
      inference::SystemSharedMemoryStatusResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::
              RequestSystemSharedMemoryStatus,
          std::mem_fn(&InferenceServiceImpl::SystemSharedMemoryStatus));
  UnaryCallData<
      inference::SystemSharedMemoryRegisterRequest,
      inference::SystemSharedMemoryRegisterResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::
              RequestSystemSharedMemoryRegister,
          std::mem_fn(&InferenceServiceImpl::SystemSharedMemoryRegister));
  UnaryCallData<
      inference::SystemSharedMemoryUnregisterRequest,
      inference::SystemSharedMemoryUnregisterResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::
              RequestSystemSharedMemoryUnregister,
          std::mem_fn(&InferenceServiceImpl::SystemSharedMemoryUnregister));
  UnaryCallData<
      inference::CudaSharedMemoryStatusRequest,
      inference::CudaSharedMemoryStatusResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::
              RequestCudaSharedMemoryStatus,
          std::mem_fn(&InferenceServiceImpl::CudaSharedMemoryStatus));
  UnaryCallData<
      inference::CudaSharedMemoryRegisterRequest,
      inference::CudaSharedMemoryRegisterResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::
              RequestCudaSharedMemoryRegister,
          std::mem_fn(&InferenceServiceImpl::CudaSharedMemoryRegister));
  UnaryCallData<
      inference::CudaSharedMemoryUnregisterRequest,
      inference::CudaSharedMemoryUnregisterResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::
              RequestCudaSharedMemoryUnregister,
          std::mem_fn(&InferenceServiceImpl::CudaSharedMemoryUnregister));
  UnaryCallData<
      inference::TraceSettingRequest, inference::TraceSettingResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::RequestTraceSetting,
          std::mem_fn(&InferenceServiceImpl::TraceSetting));
  UnaryCallData<inference::LogSettingsRequest, inference::LogSettingsResponse>::
      Start(
          async_service_, completion_queue_.get(), impl_,
          &inference::GRPCInferenceService::AsyncService::RequestLogSettings,
          std::mem_fn(&InferenceServiceImpl::LogSettings));
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
    std::unique_ptr<Server>& server)
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
  run_grpc_server_impl(service, options, server);
}

void
RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::vector<at::ScalarType>& expected_input_types,
    const std::vector<std::string>& expected_input_names,
    const std::vector<std::string>& expected_output_names,
    const GrpcServerOptions& options, std::unique_ptr<Server>& server)
{
  const GrpcModelSpec model_spec{
      .expected_input_types = expected_input_types,
      .expected_input_dims = {},
      .expected_input_names = expected_input_names,
      .expected_output_names = expected_output_names,
      .max_batch_size = 0};
  RunGrpcServer(queue, reference_outputs, model_spec, options, server);
}

void
StopServer(Server* server)
{
  if (server != nullptr) {
    set_grpc_health_status(server, false);
    server->Shutdown();
  }
}
