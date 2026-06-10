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

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "config/solver_config.hpp"
#include "convex/canonical_trace.hpp"
#include "linalg/linalg.hpp"

namespace convex {

/**
 * @brief Per-cone scalar diagnostics for one predictor or corrector pass.
 *
 * Structural fields such as cone kind, cone offset, and cone dimension are not
 * duplicated here. Join this record with `CanonicalTrace::cones[i]` when
 * producing debug output.
 */
template <linalg::FieldLike T>
struct ConeStepTrace {
  T mu{};          ///< Local normalized duality gap for this cone.
  T rhs_c_norm{};  ///< ||rhs_c||_2 in this cone block.
  T pz_rhs_norm{}; ///< ||pz_rhs||_2 after this cone contribution.
  bool active{};   ///< Set true when this cone was touched/populated.
};

/// @brief Predictor or corrector cone diagnostics for one outer iteration.
template <linalg::FieldLike T, size_t NumCones>
struct PdIpmStepTrace {
  std::array<ConeStepTrace<T>, NumCones> cones{};

  constexpr void reset() { *this = {}; }

  constexpr auto& cone(size_t i) { return cones[i]; }
  constexpr const auto& cone(size_t i) const { return cones[i]; }
};

/**
 * @brief Outer PD-IPM iteration diagnostics.
 *
 * This is intentionally global/iteration-level. Per-cone scalar diagnostics are
 * stored under `predictor` and `corrector`.
 */
template <linalg::FieldLike T, size_t NumCones>
struct PdIpmIterTrace {
  T mu{};       ///< Normalized duality gap for the full iterate.
  T alpha_x{};  ///< Max primal cone step.
  T alpha_y{};  ///< Max dual cone step.
  T rd_norm{};  ///< ||dual residual||_2.
  T rp_norm{};  ///< ||cone/primal residual||_2.
  T req_norm{}; ///< ||equality residual||_2.
  T pz_norm{};  ///< ||Newton step in z||_2.
  T pnu_norm{}; ///< ||Newton step in nu||_2.
  T px_norm{};  ///< ||Newton step in x||_2.
  T py_norm{};  ///< ||Newton step in y||_2.

  PdIpmStepTrace<T, NumCones> predictor{};
  PdIpmStepTrace<T, NumCones> corrector{};

  config::SolverResult tag{};
  linalg::NumericalEvent event{};
  bool stalled{};

  constexpr void reset() { *this = {}; }
};

/**
 * @brief Fixed-capacity PD-IPM trace.
 *
 * `num_iters` is the number of committed iterations. `current()` returns the
 * next writable slot until `commit_iter()` is called.
 */
template <linalg::FieldLike T, size_t NumCones>
struct PdIpmTrace {
  std::array<PdIpmIterTrace<T, NumCones>, PD_IPM_STEPS> iters{};
  size_t num_iters{};

  static constexpr size_t num_cones = NumCones;

  constexpr void reset() { *this = {}; }

  constexpr bool full() const { return num_iters >= PD_IPM_STEPS; }

  constexpr size_t size() const { return num_iters; }

  constexpr auto& iter(size_t k) { return iters[k]; }
  constexpr const auto& iter(size_t k) const { return iters[k]; }
  constexpr auto& operator[](size_t k) { return iters[k]; }
  constexpr const auto& operator[](size_t k) const { return iters[k]; }

  constexpr auto& current() {
    return iters[(num_iters < PD_IPM_STEPS) ? num_iters : (PD_IPM_STEPS - 1)];
  }

  constexpr const auto& current() const {
    return iters[(num_iters < PD_IPM_STEPS) ? num_iters : (PD_IPM_STEPS - 1)];
  }

  constexpr bool commit_iter() {
    if (num_iters >= PD_IPM_STEPS) return false;
    ++num_iters;
    return true;
  }
};

/// @brief Empty object used when full snapshot tracing is disabled.
struct EmptyPdIpmSnapshot {};

template <linalg::FieldLike T, size_t MaxConeDim>
struct PdIpmConeSnapshot {
  std::array<T, MaxConeDim> xk{};
  std::array<T, MaxConeDim> yk{};
  std::array<T, MaxConeDim> rhs_c{};
  std::array<T, MaxConeDim> px{};
  std::array<T, MaxConeDim> py{};
};

template <linalg::FieldLike T, size_t VarDim, size_t EqDim>
struct PdIpmGlobalStepSnapshot {
  std::array<T, VarDim> pz{};
  std::array<T, EqDim> pnu{};
};

/**
 * @brief Optional full debug snapshot for one PD-IPM iteration.
 *
 * This stores cone-local snapshots per cone, and stores global `pz`/`pnu` once
 * per predictor/corrector pass rather than duplicating them per cone.
 */
template <linalg::FieldLike T, size_t NumCones, size_t MaxConeDim,
          size_t VarDim, size_t EqDim>
struct PdIpmFullIterSnapshot {
  std::array<PdIpmConeSnapshot<T, MaxConeDim>, NumCones> predictor_cones{};
  std::array<PdIpmConeSnapshot<T, MaxConeDim>, NumCones> corrector_cones{};
  PdIpmGlobalStepSnapshot<T, VarDim, EqDim> predictor_global{};
  PdIpmGlobalStepSnapshot<T, VarDim, EqDim> corrector_global{};
};

template <bool Enable, linalg::FieldLike T, size_t NumCones, size_t MaxConeDim,
          size_t VarDim, size_t EqDim>
using PdIpmSnapshotStorage = std::conditional_t<
    Enable, PdIpmFullIterSnapshot<T, NumCones, MaxConeDim, VarDim, EqDim>,
    EmptyPdIpmSnapshot>;

/// @brief Convenience alias for a solver trace matched to a CanonicalTrace.
template <linalg::FieldLike T, typename CanonicalTraceT>
using PdIpmTraceForCanonicalTrace =
    PdIpmTrace<T, canonical_trace_num_cones_v<CanonicalTraceT>>;

} // namespace convex
