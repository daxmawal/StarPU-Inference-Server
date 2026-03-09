#pragma once

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using ResolvePythonCandidatesOverrideForTestFn =
    std::vector<std::filesystem::path> (*)();

using ResolvePythonIsRegularFileOverrideForTestFn =
    bool (*)(const std::filesystem::path&, std::error_code&);

auto
resolve_python_candidates_override_for_test() noexcept
    -> ResolvePythonCandidatesOverrideForTestFn&
{
  static ResolvePythonCandidatesOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
resolve_python_is_regular_file_override_for_test() noexcept
    -> ResolvePythonIsRegularFileOverrideForTestFn&
{
  static ResolvePythonIsRegularFileOverrideForTestFn override_fn = nullptr;
  return override_fn;
}
#endif  // SONAR_IGNORE_END
