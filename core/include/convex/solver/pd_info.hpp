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

#include "convex/solver_traits.hpp"

namespace convex {

/**
 * @brief PrimalDualIterate
 */
template <linalg::VectorLike Var, linalg::VectorLike ConeVar,
          linalg::VectorLike NuVar>
struct PrimalDualIterate {
  Var z{};     ///< decision var in Rn + aux lifts
  ConeVar x{}; ///< primal cone var in K
  ConeVar y{}; ///< dual cone var in K^*
  NuVar nu{};  ///< lagrangian multiplier on equality constraints
};

/**
 * @brief PrimalDualResult
 */
template <linalg::VectorLike Var, linalg::VectorLike ConeVar,
          linalg::VectorLike NuVar>
struct PrimalDualResult {
  using T = linalg::vec_type_t<Var>;

  config::SolverResult tag{}; ///< tag for result of pd-ipm
  linalg::NumericalEvent event{
      linalg::NumericalEvent::NONE}; ///< event status (triggers on numerical
                                     ///< detection or failure)
  bool stalled{false};               ///< backtracking status (stalled or not)
  T mu{};                            ///< normalized duality gap
  T convex_obj{};                    ///< convex objective value c^T z
  T dual_obj{};                      ///< dual objective value -g^T y
  Var rd{};                          ///< dual residuals
  ConeVar rp{};                      ///< primal residuals
  NuVar req{};                       ///< equality residuals
  PrimalDualIterate<Var, ConeVar, NuVar> iterate; ///< primal dual iterate
  size_t num_iter{};                              ///< total number of iter
};

} // namespace convex
