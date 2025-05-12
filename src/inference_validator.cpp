#include "inference_validator.hpp"

#include <iostream>

bool
validate_outputs(
    const at::Tensor& output_direct, const at::Tensor& output_starpu,
    double tolerance)
{
  at::Tensor diff = torch::abs(output_direct - output_starpu);
  double max_diff = diff.max().item<double>();
  std::cout << "Max difference with direct inference: " << max_diff
            << std::endl;

  if (max_diff < tolerance) {
    std::cout << "StarPU output matches direct inference (within tolerance).\n";
    return true;
  } else {
    std::cerr << "Mismatch between StarPU and direct inference! Max diff = "
              << max_diff << "\n";
    return false;
  }
}