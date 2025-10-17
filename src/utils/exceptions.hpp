#pragma once

#include <stdexcept>
#include <string>

namespace starpu_server {
// =============================================================================
// Base class for all inference-related exceptions
// =============================================================================

class InferenceEngineException : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// =============================================================================
// Specific exception types for common failure cases
// =============================================================================

/// Thrown when an unsupported data type is used
class UnsupportedDtypeException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

/// Thrown when inference execution fails
class InferenceExecutionException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

/// Thrown when a StarPU registration fails
class StarPURegistrationException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

/// Thrown when registering output buffers with StarPU fails
class OutputSlotRegistrationError : public StarPURegistrationException {
 public:
  using StarPURegistrationException::StarPURegistrationException;
};

/// Thrown when an invalid inference job is submitted or accessed
class InvalidInferenceJobException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

/// Thrown when a memory allocation fails
class MemoryAllocationException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

/// Thrown when a StarPU task cannot be submitted
class StarPUTaskSubmissionException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

/// Thrown when a StarPU task cannot be created
class StarPUTaskCreationException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

/// Thrown when the model output type is not supported
class UnsupportedModelOutputTypeException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

/// Thrown when loading the model or its reference outputs fails
class ModelLoadingException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

class TooManyGpuModelsException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

/// Thrown when an invalid GPU device identifier is provided
class InvalidGpuDeviceException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

/// Thrown when execution of a StarPU codelet fails
class StarPUCodeletException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

/// Thrown when StarPU initialization fails
class StarPUInitializationException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

/// Thrown when retrieving CUDA workers fails
class StarPUWorkerQueryException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

/// Thrown when computing the message size would overflow
class MessageSizeOverflowException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

/// Thrown when a tensor dimension is negative
class InvalidDimensionException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

}  // namespace starpu_server
