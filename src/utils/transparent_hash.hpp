#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>

struct TransparentHash {
  using is_transparent = void;

  auto operator()(std::string_view key) const noexcept -> std::size_t
  {
    return std::hash<std::string_view>{}(key);
  }

  auto operator()(const std::string& key) const noexcept -> std::size_t
  {
    return std::hash<std::string_view>{}(key);
  }

  auto operator()(const char* key) const noexcept -> std::size_t
  {
    return std::hash<std::string_view>{}(key);
  }
};
