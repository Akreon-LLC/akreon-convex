#pragma once

#include <akreon/linalg/test_helpers.hpp>
#include <akreon/convex/debug/pd_ipm_format.hpp>
#include <iostream>

namespace unit_test {
/**
 * @brief Print a solve result and its attached canonicalization/solver traces.
 *
 * @param res Primal-dual solve result, including any recorded traces.
 */
inline void log(const auto& res) {
  char buf[64000];
  convex::debug::result_to_cstr(res, buf, sizeof(buf));
  std::cout << buf << std::endl;
}

} // namespace unit_test
