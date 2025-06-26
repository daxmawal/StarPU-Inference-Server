#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>

struct TransparentHash {
  using is_transparent = void;

  std::size_t operator()(std::string_view key) const noexcept
  {
    return std::hash<std::string_view>{}(key);
  }

  std::size_t operator()(const std::string& key) const noexcept
  {
    return std::hash<std::string_view>{}(key);
  }

  std::size_t operator()(const char* key) const noexcept
  {
    return std::hash<std::string_view>{}(key);
  }
};
