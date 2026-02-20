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
struct SolverTraits {};

template <>
struct SolverTraits<float> {
  static constexpr size_t STEPS = 32;
  static constexpr float TOL = 1e-8;
  static constexpr double PRIMAL_FEASIBLE = 1e-5;
  static constexpr double DUAL_FEASIBLE = 1e-5;
  static constexpr double DUALITY_GAP = 1e-4;
};

template <>
struct SolverTraits<double> {
  static constexpr size_t STEPS = 64;
  static constexpr float TOL = 1e-8;
  static constexpr double PRIMAL_FEASIBLE = 1e-5;
  static constexpr double DUAL_FEASIBLE = 1e-5;
  static constexpr double DUALITY_GAP = 1e-4;
};

enum struct SolverResult : uint8_t {
  INVALID = 0,
  OPTIMAL,
  PRIMAL_FEASIBLE,
  DUAL_FEASIBLE,
  PRIMAL_DUAL_FEASIBLE,
  INFEASIBLE_OR_UNBOUNDED
};

}  // namespace linalg
