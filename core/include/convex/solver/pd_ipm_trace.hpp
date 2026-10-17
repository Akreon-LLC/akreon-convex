/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
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
template <linalg::Field T>
struct ConeStepTrace {
  T mu{};          ///< Local normalized duality gap for this cone.
  T rhs_c_norm{};  ///< ||rhs_c||_2 in this cone block.
  T pz_rhs_norm{}; ///< ||pz_rhs||_2 after this cone contribution.
  bool active{};   ///< Set true when this cone was touched/populated.
};

/// @brief Predictor or corrector cone diagnostics for one outer iteration.
template <linalg::Field T, std::size_t NumCones>
struct PdIpmStepTrace {
  /**
   * @brief reset
   *
   * @return void
   */
  constexpr void reset() { *this = {}; }

  /**
   * @brief cone
   *
   * @param idx
   *
   * @return cone at index
   */
  constexpr auto& cone(std::size_t idx) {
    AK_ASSERT(idx < NumCones);
    return cones[idx];
  }
  constexpr const auto& cone(std::size_t idx) const {
    AK_ASSERT(idx < NumCones);
    return cones[idx];
  }

  /** @brief public member vars */
  std::array<ConeStepTrace<T>, NumCones> cones{}; ///< Array of cone step traces
};

/**
 * @brief Outer PD-IPM iteration diagnostics.
 *
 * This is intentionally global/iteration-level. Per-cone scalar diagnostics are
 * stored under `predictor` and `corrector`.
 */
template <linalg::Field T, std::size_t NumCones>
struct PdIpmIterTrace {

  /**
   * @brief reset
   *
   * @return void
   */
  constexpr void reset() { *this = {}; }

  /** @brief public member vars */
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

  PdIpmStepTrace<T, NumCones> predictor{}; ///< predictor pd ipm trace
  PdIpmStepTrace<T, NumCones> corrector{}; ///< corrector pd ipm trace

  config::SolverResult tag{};     ///< tag of solver result
  linalg::NumericalEvent event{}; ///< event of reported numerical result
  bool stalled{};                 ///< tag whether or not its been stalled
};

/**
 * @brief Fixed-capacity PD-IPM trace.
 *
 * `num_iters` is the number of committed iterations. `current()` returns the
 * next writable slot until `commit_iter()` is called.
 */
template <linalg::Field T, std::size_t NumCones>
struct PdIpmTrace {
  constexpr void reset() { *this = {}; }
  /// @brief static export of num cones
  static constexpr std::size_t num_cones = NumCones;

  /**
   * @brief full (num_iters >= PD_IPM_STEPS)
   *
   * @return T/F
   */
  constexpr bool full() const { return num_iters >= PD_IPM_STEPS; }

  /**
   * @brief size
   *
   * @return num_iters
   */
  constexpr std::size_t size() const { return num_iters; }

  /**
   * @brief iter & relevent overrides
   *
   * @param k (std::size_t)
   *
   * @return iter at k
   */
  constexpr auto& iter(std::size_t k) { return iters[k]; }
  constexpr const auto& iter(std::size_t k) const { return iters[k]; }
  constexpr auto& operator[](std::size_t k) { return iters[k]; }
  constexpr const auto& operator[](std::size_t k) const { return iters[k]; }

  /**
   * @brief current
   *
   * @return current iter reference
   */
  constexpr auto& current() {
    return iters[(num_iters < PD_IPM_STEPS) ? num_iters : (PD_IPM_STEPS - 1)];
  }

  /**
   * @brief current
   *
   * @return current iter
   */
  constexpr const auto& current() const {
    return iters[(num_iters < PD_IPM_STEPS) ? num_iters : (PD_IPM_STEPS - 1)];
  }

  /**
   * @brief commit iter will increment the number of iters
   *
   * @return T/F
   */
  constexpr bool commit_iter() {
    if (num_iters >= PD_IPM_STEPS) return false;
    ++num_iters;
    return true;
  }

  /** @brief public member vars */
  std::array<PdIpmIterTrace<T, NumCones>, PD_IPM_STEPS>
      iters{};             ///< array of pd ipm trace objects
  std::size_t num_iters{}; ///< number of iterations
};

/// @brief Empty object used when full snapshot tracing is disabled.
struct EmptyPdIpmSnapshot {};

/**
 * @brief Cone snap shot at current iterate
 */
template <linalg::Field T, std::size_t MaxConeDim>
struct PdIpmConeSnapshot {
  std::array<T, MaxConeDim> xk{};    ///< primal var at k
  std::array<T, MaxConeDim> yk{};    ///< dual var at k
  std::array<T, MaxConeDim> rhs_c{}; ///< rhs at k in pred-corr
  std::array<T, MaxConeDim> px{};    ///< primal newton step
  std::array<T, MaxConeDim> py{};    ///< dual newton step
};

/**
 * @brief Global snap shot at current iterate
 */
template <linalg::Field T, std::size_t VarDim, std::size_t EqDim>
struct PdIpmGlobalStepSnapshot {
  std::array<T, VarDim> pz{}; ///< dec var newton step
  std::array<T, EqDim> pnu{}; ///< equality newton step
};

/**
 * @brief Optional full debug snapshot for one PD-IPM iteration.
 *
 * This stores cone-local snapshots per cone, and stores global `pz`/`pnu` once
 * per predictor/corrector pass rather than duplicating them per cone.
 */
template <linalg::Field T, std::size_t NumCones, std::size_t MaxConeDim,
          std::size_t VarDim, std::size_t EqDim>
struct PdIpmFullIterSnapshot {
  std::array<PdIpmConeSnapshot<T, MaxConeDim>, NumCones>
      predictor_cones{}; ///< predictor cone snap shot
  std::array<PdIpmConeSnapshot<T, MaxConeDim>, NumCones>
      corrector_cones{}; ///< corrector cone snap shot
  PdIpmGlobalStepSnapshot<T, VarDim, EqDim>
      predictor_global{}; ///< predicator global snap shot
  PdIpmGlobalStepSnapshot<T, VarDim, EqDim>
      corrector_global{}; ///< corrector global snap shot
};

/// @brief Alias for snap shot storage (full or empty)
template <bool Enable, linalg::Field T, std::size_t NumCones,
          std::size_t MaxConeDim, std::size_t VarDim, std::size_t EqDim>
using PdIpmSnapshotStorage = std::conditional_t<
    Enable, PdIpmFullIterSnapshot<T, NumCones, MaxConeDim, VarDim, EqDim>,
    EmptyPdIpmSnapshot>;

/// @brief Convenience alias for a solver trace matched to a CanonicalTrace.
template <linalg::Field T, typename CanonicalTraceT>
using PdIpmTraceForCanonicalTrace =
    PdIpmTrace<T, canonical_trace_num_cones_v<CanonicalTraceT>>;

} // namespace convex
