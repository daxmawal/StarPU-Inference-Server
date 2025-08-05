#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <unordered_map>

#include "utils/transparent_hash.hpp"

TEST(TransparentHash, TransparentLookup)
{
  std::unordered_map<std::string, int, TransparentHash, std::equal_to<>> map;
  map.emplace("apple", 1);
  map.emplace(std::string{"banana"}, 2);
  std::string str_key = "apple";
  std::string_view sv_key = "banana";
  const char* c_key = "apple";
  auto iter1 = map.find(str_key);
  ASSERT_NE(iter1, map.end());
  EXPECT_EQ(iter1->second, 1);
  auto iter2 = map.find(sv_key);
  ASSERT_NE(iter2, map.end());
  EXPECT_EQ(iter2->second, 2);
  auto iter3 = map.find(c_key);
  ASSERT_NE(iter3, map.end());
  EXPECT_EQ(iter3->second, 1);
}
