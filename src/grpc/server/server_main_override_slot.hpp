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

#define STARPU_SERVER_DEFINE_TEST_OVERRIDE_SLOT(accessor_name, fn_type)      \
  auto accessor_name() noexcept -> fn_type&                                  \
  {                                                                          \
    struct accessor_name##Tag;                                               \
    return ::starpu_server::testing::server_main::detail::override_slot_ref< \
        accessor_name##Tag, fn_type>();                                      \
  }
#endif  // defined(STARPU_TESTING)
