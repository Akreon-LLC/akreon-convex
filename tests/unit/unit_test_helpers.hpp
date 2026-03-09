/*
 * Copyright © 2025 Akreon LLC. All rights reserved.
 *
 * Proprietary and Confidential.
 * This source code is the property of Akreon LLC and constitutes a trade secret
 * protected by U.S. and international law.
 *
 * Unauthorized copying, modification, distribution, or disclosure of this file,
 * via any medium, is strictly prohibited without express written permission
 * from Akreon LLC.
 */

#include <iostream>

namespace unit_test {

constexpr double eps = 1e-8;

inline constexpr void log(const auto& res, const auto& trace) {
  char buf[64000];
  trace_to_cstr(res.num_iter, trace, buf, sizeof(buf));
  std::cout << buf << std::endl;

  memset(buf, '\0', sizeof(buf));
  result_to_cstr(res, buf, sizeof(buf));
  std::cout << buf << std::endl;
}

}  // namespace unit_test
