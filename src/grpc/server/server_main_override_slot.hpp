#pragma once

#if defined(STARPU_TESTING)
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

}  // namespace starpu_server::testing::server_main::detail
#endif  // defined(STARPU_TESTING)
