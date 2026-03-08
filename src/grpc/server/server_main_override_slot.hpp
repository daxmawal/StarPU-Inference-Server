#pragma once

#if defined(STARPU_TESTING)
#include <functional>
#include <type_traits>
#include <utility>

namespace starpu_server::testing::server_main::detail {

template <typename T>
class OverrideSlot {
 public:
  constexpr OverrideSlot() noexcept = default;

  auto ref() noexcept -> T& { return value_; }

 private:
  T value_{};
};

template <typename Tag, typename T>
auto
override_slot_ref() noexcept -> T&
{
  static OverrideSlot<T> slot{};
  return slot.ref();
}

template <typename OverrideAccessor, typename FallbackFn, typename... Args>
auto
call_override_or(
    OverrideAccessor&& override_accessor, FallbackFn&& fallback_fn,
    Args&&... args) -> std::invoke_result_t<FallbackFn, Args...>
  requires(!std::is_void_v<std::invoke_result_t<FallbackFn, Args...>>)
{
  using Return = std::invoke_result_t<FallbackFn, Args...>;
  if (const auto override_fn =
          std::forward<OverrideAccessor>(override_accessor)();
      override_fn != nullptr) {
    return static_cast<Return>(
        std::invoke(override_fn, std::forward<Args>(args)...));
  }

  return std::invoke(
      std::forward<FallbackFn>(fallback_fn), std::forward<Args>(args)...);
}

template <typename OverrideAccessor, typename FallbackFn, typename... Args>
void
call_override_or(
    OverrideAccessor&& override_accessor, FallbackFn&& fallback_fn,
    Args&&... args)
  requires std::is_void_v<std::invoke_result_t<FallbackFn, Args...>>
{
  if (const auto override_fn =
          std::forward<OverrideAccessor>(override_accessor)();
      override_fn != nullptr) {
    std::invoke(override_fn, std::forward<Args>(args)...);
    return;
  }
  std::invoke(
      std::forward<FallbackFn>(fallback_fn), std::forward<Args>(args)...);
}

}  // namespace starpu_server::testing::server_main::detail

#define STARPU_SERVER_DEFINE_TEST_OVERRIDE_SLOT(accessor_name, fn_type)      \
  auto accessor_name() noexcept -> fn_type&                                  \
  {                                                                          \
    struct accessor_name##Tag;                                               \
    return ::starpu_server::testing::server_main::detail::override_slot_ref< \
        accessor_name##Tag, fn_type>();                                      \
  }

#define STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT( \
    alias_name, accessor_name, ...)               \
  using alias_name = __VA_ARGS__;                 \
  STARPU_SERVER_DEFINE_TEST_OVERRIDE_SLOT(accessor_name, alias_name)
#endif  // defined(STARPU_TESTING)
