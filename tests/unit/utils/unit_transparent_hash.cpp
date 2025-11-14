#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <unordered_map>

#include "utils/transparent_hash.hpp"

TEST(TransparentHash_Unit, TransparentLookup)
{
  std::unordered_map<std::string, int, TransparentHash, std::equal_to<>> map;
  map.try_emplace("apple", 1);
  map.try_emplace(std::string{"banana"}, 2);

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

TEST(TransparentHash_Unit, LookupMissingKeysReturnsEnd)
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

TEST(TransparentHash_Unit, HeterogeneousEraseAndCount)
{
  std::unordered_map<std::string, int, TransparentHash, std::equal_to<>> map;
  map.emplace("a", 1);
  map.emplace("bb", 2);
  map.emplace("ccc", 3);

  map.erase(std::string{"a"});
  map.erase(std::string{"bb"});

  EXPECT_EQ(map.size(), 1U);
  EXPECT_EQ(map.count(std::string{"ccc"}), 1U);
  EXPECT_EQ(map.count(std::string{"bb"}), 0U);
  EXPECT_EQ(map.count("a"), 0U);
}
