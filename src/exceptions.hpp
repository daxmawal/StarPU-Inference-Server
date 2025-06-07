#pragma once

#include <stdexcept>
#include <string>

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
  explicit UnsupportedDtypeException(const std::string& msg)
      : InferenceEngineException(msg)
  {
  }
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
