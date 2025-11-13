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

#pragma once

#include <cstddef>

namespace linalg {

template <typename T>
struct SolverTraits {
  static constexpr size_t steps = 0;
  static constexpr T tol = T(0);
};

template <>
struct SolverTraits<float> {
  static constexpr size_t steps = 16;
  static constexpr float tol = 1e-8;
};

template <>
struct SolverTraits<double> {
  static constexpr size_t steps = 64;
  static constexpr float tol = 1e-10;
};

}  // namespace linalg
