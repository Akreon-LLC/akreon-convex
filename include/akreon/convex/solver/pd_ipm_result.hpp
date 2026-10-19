/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <type_traits>

#include "akreon/convex/solver_config.hpp"
#include "akreon/convex/canonical_trace.hpp"
#include "akreon/convex/solver/pd_info.hpp"
#include "akreon/convex/solver/pd_ipm_trace.hpp"
#include "akreon/convex/solver_traits.hpp"
#include "akreon/linalg/linalg.hpp"

namespace convex {

/**
 * @brief Empty solver trace for users/builds that only want the final result.
 */
struct EmptySolverTrace {
  std::size_t num_iters{};
  constexpr void reset() { *this = {}; }
};

/**
 * @brief Final primal-dual solve result plus optional structural and iteration
 * traces.
 */
template <linalg::VectorContainer Var, linalg::VectorContainer ConeVar,
          linalg::VectorContainer NuVar,
          typename CanonicalTraceT = EmptyCanonicalTrace,
          typename SolverTraceT = EmptySolverTrace>
struct PrimalDualResult {
  using Scalar = linalg::vector_element_t<Var>;
  using CanonicalTrace = CanonicalTraceT;
  using SolverTrace = SolverTraceT;

  config::SolverResult tag{};     ///< Solver result tag
  linalg::NumericalEvent event{}; ///< Report of any numerical events
  bool stalled{}; ///< Tag that indicates if backtracking stalled
  std::size_t
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
