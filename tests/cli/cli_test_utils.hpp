#pragma once

#include <initializer_list>
#include <vector>

inline std::vector<char*>
build_argv(std::initializer_list<const char*> args)
{
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (const char* arg : args) {
    argv.push_back(const_cast<char*>(arg));
  }
  return argv;
}
