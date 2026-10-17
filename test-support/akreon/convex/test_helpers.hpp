#pragma once
#include <akreon/linalg/test_helpers.hpp>
#include <akreon/convex/debug/pd_ipm_format.hpp>
namespace unit_test {
/**
 * @brief log
 *
 * @param res
 * @param trace
 *
 * @return void
 */
inline constexpr void log(const auto& res, const auto& trace) {
  static_cast<void>(trace);
  char buf[64000];
  convex::debug::result_to_cstr(res, buf, sizeof(buf));
  std::cout << buf << std::endl;
}

} // namespace unit_test
