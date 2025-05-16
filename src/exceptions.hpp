#pragma once
#include <stdexcept>
#include <string>

class InferenceEngineException : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class UnsupportedDtypeException : public std::runtime_error {
 public:
  explicit UnsupportedDtypeException(const std::string& msg)
      : std::runtime_error(msg)
  {
  }
};

class InferenceExecutionException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

class StarPURegistrationException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

class InvalidInferenceJobException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

class MemoryAllocationException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};

class StarPUTaskSubmissionException : public InferenceEngineException {
 public:
  using InferenceEngineException::InferenceEngineException;
};