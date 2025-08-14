#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <unordered_map>

#include "utils/transparent_hash.hpp"

TEST(TransparentHash_Robustesse, LookupMissingKeysReturnsEnd)
{
  std::unordered_map<std::string, int, TransparentHash, std::equal_to<>> map;
  map.emplace("apple", 1);

  std::string no_str = "pear";
  std::string_view no_sv = "peach";
  const char* no_c = "plum";

  EXPECT_EQ(map.find(no_str), map.end());
  EXPECT_EQ(map.find(no_sv), map.end());
  EXPECT_EQ(map.find(no_c), map.end());
}

TEST(TransparentHash_Robustesse, HeterogeneousEraseAndCount)
{
  std::unordered_map<std::string, int, TransparentHash, std::equal_to<>> map;
  map.emplace("a", 1);
  map.emplace("bb", 2);
  map.emplace("ccc", 3);

  EXPECT_EQ(map.size(), 1u);
  EXPECT_EQ(map.count(std::string{"ccc"}), 1u);
  EXPECT_EQ(map.count(std::string_view{"bb"}), 0u);
  EXPECT_EQ(map.count("a"), 0u);
}
