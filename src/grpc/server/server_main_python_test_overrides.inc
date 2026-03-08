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
#endif  // SONAR_IGNORE_STOP

auto
resolve_python_candidates_for_runtime(
    std::span<const std::filesystem::path> default_candidates,
    std::vector<std::filesystem::path>& override_candidates_storage)
    -> std::span<const std::filesystem::path>
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = resolve_python_candidates_override_for_test();
      override_fn != nullptr) {
    override_candidates_storage = override_fn();
    return override_candidates_storage;
  }
#else
  (void)override_candidates_storage;
#endif  // SONAR_IGNORE_STOP
  return default_candidates;
}

auto
resolve_python_is_regular_file_for_runtime(
    const std::filesystem::path& candidate, std::error_code& status_ec) -> bool
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = resolve_python_is_regular_file_override_for_test();
      override_fn != nullptr) {
    return override_fn(candidate, status_ec);
  }
#endif  // SONAR_IGNORE_STOP
  return std::filesystem::is_regular_file(candidate, status_ec);
}
