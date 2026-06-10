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
#include <type_traits>

#include "config/solver_config.hpp"
#include "convex/canonical_trace.hpp"
#include "convex/solver/pd_info.hpp"
#include "convex/solver/pd_ipm_trace.hpp"
#include "convex/solver_traits.hpp"
#include "linalg/linalg.hpp"

namespace convex {

/**
 * @brief Empty solver trace for users/builds that only want the final result.
 */
struct EmptySolverTrace {
  size_t num_iters{};
  constexpr void reset() { *this = {}; }
};

/**
 * @brief Final primal-dual solve result plus optional structural and iteration
 * traces.
 */
template <linalg::VectorLike Var, linalg::VectorLike ConeVar,
          linalg::VectorLike NuVar,
          typename CanonicalTraceT = EmptyCanonicalTrace,
          typename SolverTraceT = EmptySolverTrace>
struct PrimalDualResult {
  using Scalar = linalg::vec_type_t<Var>;
  using CanonicalTrace = CanonicalTraceT;
  using SolverTrace = SolverTraceT;

  config::SolverResult tag{};     ///< Solver result tag
  linalg::NumericalEvent event{}; ///< Report of any numerical events
  bool stalled{}; ///< Tag that indicates if backtracking stalled
  size_t
      num_iter{}; ///< Number of iterations until convergence, stalled, or max

  Scalar mu{};         ///< Normalized duality gap
  Scalar convex_obj{}; ///< Convex objective result
  Scalar dual_obj{};   ///< Dual objective result

  PrimalDualIterate<Var, ConeVar, NuVar>
      iterate{}; ///< PD iterate (z, x, y, nu)
  Var rd{};      ///< Dual residual
  ConeVar rp{};  ///< Primal residual
  NuVar req{};   ///< Equality residual

  CanonicalTraceT canonical_trace{}; ///< Attached canonical trace
  SolverTraceT solver_trace{};       ///< Solver trace
};

} // namespace convex
