#include "time_utils.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

namespace time_utils {

auto
format_timestamp(const std::chrono::high_resolution_clock::time_point&
                     time_point) -> std::string
{
  constexpr int MillisecondsPerSecond = 1000;
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                          time_point.time_since_epoch()) %
                      MillisecondsPerSecond;

  std::time_t time = std::chrono::system_clock::to_time_t(
      std::chrono::time_point_cast<std::chrono::system_clock::duration>(
          time_point));
  std::tm local_tm{};
  localtime_r(&time, &local_tm);
  std::ostringstream oss;
  oss << std::put_time(&local_tm, "%H:%M:%S") << '.' << std::setfill('0')
      << std::setw(3) << milliseconds.count();
  return oss.str();
}

}  // namespace time_utils