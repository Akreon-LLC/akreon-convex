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

#include "cone_traits.hpp"
#include "convex_traits.hpp"
#include "linalg/linalg_util.hpp"
#ifdef AKREON_DEBUG
#include "pd_ipm_debug.hpp"
#endif

namespace convex {

/**
 * @brief Predictor Corrector
 */
template <size_t NumCones, linalg::MatrixLike Mat, linalg::VectorLike Var,
          linalg::VectorLike ConeVar>
struct PredictorCorrector {
  /// @brief extract T and N from Mat
  using T = linalg::mat_type_t<Mat>;
  static constexpr size_t N = linalg::mat_cols_v<Mat>;

  /**
   * @brief predictor (affine-scaling) direction
   *
   * @param pz_rhs (rhs forward pass accumulate)
   * @param K (cartesian product of cones)
   * @param S_inv (factored)
   * @param F (R^n -> conic space)
   * @param rp (primal residual)
   * @param rd (dual residual)
   * @param x (x \in K iterate)
   * @param y (y \in K^* iterate)
   * @param pz (newton step in z)
   * @param px (newton step in x)
   * @param py (newton step in y)
   *
   * @return pz_rhs (ignore on return)
   */
  template <size_t idx = 0, size_t Start = 0>
  constexpr auto predictor(
      Var&& pz_rhs, const auto& K, const auto& S_inv, const Mat& F,
      const ConeVar& rp, const Var& rd, const ConeVar& x, const ConeVar& y,
      Var& pz, ConeVar& px, ConeVar& py
#ifdef AKREON_DEBUG
      ,
      IterState<T, NumCones, linalg::mat_rows_v<Mat>, N>& trace_state
#endif
  ) const {
    if constexpr (idx >= NumCones)
      return std::tuple(std::move(pz_rhs), linalg::NumericalEvent::NONE);
    else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;
      constexpr size_t KiRows = linalg::vec_size_v<Slack>;
      constexpr size_t End = Start + KiRows;

      using ConePoint = linalg::Vector<T, End - Start>;
      ConePoint xk{}, yk{}, pxk{}, pyk{}, rpk{};

      for (size_t i = Start, j = 0; i < End; ++i, ++j) {
        xk(j) = x(i);
        yk(j) = y(i);
        pxk(j) = px(i);
        pyk(j) = py(i);
        rpk(j) = rp(i);
      }

      const auto [Hes, event] = Ki.H(xk, yk);
      if (linalg::is_fatal_event(event))
        return std::tuple(std::move(pz_rhs), event);

      const auto F_blk = linalg::get_block<Start, End, 0, N>(F);
      const auto F_blk_T = linalg::transpose(F_blk);

      ConePoint rhs_c = yk;

      pz_rhs = pz_rhs - F_blk_T * Hes.apply(rpk) - F_blk_T * rhs_c;
#ifdef AKREON_DEBUG
      auto& cone_trace = trace_state.predictor[idx];
      cone_trace.mu = Ki.trace(xk, yk) / Ki.nu();
      cone_trace.rhs_c = linalg::L2<ConePoint>()(rhs_c);
      cone_trace.pz_rhs = linalg::L2<Var>()(pz_rhs);
      cone_trace.cone_dim = End - Start;
      for (size_t j = 0; j < End - Start; ++j) {
        cone_trace.xk[j] = xk(j);
        cone_trace.yk[j] = yk(j);
        cone_trace.rhs_c_vec[j] = rhs_c(j);
      }
      cone_trace.cone_type = detail::cone_type_name_v<decltype(Ki)>;
#endif

      auto [new_pz_rhs, next_event] = predictor<idx + 1, End>(
          std::move(pz_rhs), K, S_inv, F, rp, rd, x, y, pz, px, py
#ifdef AKREON_DEBUG
          ,
          trace_state
#endif
      );
      if (linalg::is_fatal_event(next_event))
        return std::tuple(std::move(new_pz_rhs), next_event);

      if constexpr (idx + 1 == NumCones) {
        const auto [pz_next, solve_event] = S_inv.solve(new_pz_rhs + rd);
        if (linalg::is_fatal_event(solve_event))
          return std::tuple(std::move(new_pz_rhs), solve_event);
        pz = pz_next;
      }

      const auto px_local = F_blk * pz + rpk;
      const auto py_local = T(-1) * rhs_c - Hes.apply(px_local);

      for (size_t i = Start, j = 0; i < End; ++i, ++j) {
        px(i) = px_local(j), py(i) = py_local(j);
      }
#ifdef AKREON_DEBUG
      for (size_t j = 0; j < N; ++j) cone_trace.pz[j] = pz(j);
      for (size_t j = 0; j < End - Start; ++j) {
        cone_trace.px[j] = px_local(j);
        cone_trace.py[j] = py_local(j);
      }
#endif

      return std::tuple(std::move(new_pz_rhs), linalg::NumericalEvent::NONE);
    }
  }

  /**
   * @brief corrector (centering) direction
   *
   * @param pz_rhs (forward pass accumulate)
   * @param px_pred (newton step affine scaling dir x)
   * @param py_pred (newton step affine scaling dir y)
   * @param K (cartesian product of cones)
   * @param S_inv (factored)
   * @param F (R^n -> conic space)
   * @param rp (primal residual)
   * @param rd (dual residual)
   * @param x (x \in K iterate)
   * @param y (y \in K^* iterate)
   * @param pz (newton step in z)
   * @param px (newton step in x)
   * @param py (newton step in y)
   *
   * @return pz_rhs (ignore on return)
   */
  template <size_t idx = 0, size_t Start = 0>
  constexpr auto corrector(
      Var&& pz_rhs, const ConeVar& px_aff, const ConeVar& py_aff, const auto& K,
      const auto& S_inv, const Mat& F, const ConeVar& rp, const Var& rd,
      const ConeVar& x, const ConeVar& y, Var& pz, ConeVar& px, ConeVar& py
#ifdef AKREON_DEBUG
      ,
      IterState<T, NumCones, linalg::mat_rows_v<Mat>, N>& trace_state
#endif
  ) const {
    if constexpr (idx >= NumCones)
      return std::tuple(std::move(pz_rhs), linalg::NumericalEvent::NONE);
    else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;
      constexpr size_t KiRows = linalg::vec_size_v<Slack>;
      constexpr size_t End = Start + KiRows;

      using ConePoint = linalg::Vector<T, End - Start>;
      ConePoint xk{}, yk{}, px_aff_k{}, py_aff_k{}, rpk{};

      for (size_t i = Start, j = 0; i < End; ++i, ++j) {
        xk(j) = x(i);
        yk(j) = y(i);
        px_aff_k(j) = px_aff(i);
        py_aff_k(j) = py_aff(i);
        rpk(j) = rp(i);
      }

      const auto [Hes, event] = Ki.H(xk, yk);
      if (linalg::is_fatal_event(event))
        return std::tuple(std::move(pz_rhs), event);

      const auto F_blk = linalg::get_block<Start, End, 0, N>(F);
      const auto F_blk_T = linalg::transpose(F_blk);

      T mu_k = Ki.trace(xk, yk) / Ki.nu();
      const auto [grad_xk, grad_event] = Ki.grad(xk);
      if (linalg::is_fatal_event(grad_event))
        return std::tuple(std::move(pz_rhs), grad_event);

      ConePoint rhs_c = yk + mu_k * grad_xk + Ki.circ(px_aff_k, py_aff_k);

      pz_rhs = pz_rhs - F_blk_T * Hes.apply(rpk) - F_blk_T * rhs_c;
#ifdef AKREON_DEBUG
      auto& cone_trace = trace_state.corrector[idx];
      cone_trace.mu = mu_k;
      cone_trace.rhs_c = linalg::L2<ConePoint>()(rhs_c);
      cone_trace.pz_rhs = linalg::L2<Var>()(pz_rhs);
      cone_trace.cone_dim = End - Start;
      for (size_t j = 0; j < End - Start; ++j) {
        cone_trace.xk[j] = xk(j);
        cone_trace.yk[j] = yk(j);
        cone_trace.rhs_c_vec[j] = rhs_c(j);
      }
      cone_trace.cone_type = detail::cone_type_name_v<decltype(Ki)>;
#endif

      auto [new_pz_rhs, next_event] =
          corrector<idx + 1, End>(std::move(pz_rhs), px_aff, py_aff, K, S_inv,
                                  F, rp, rd, x, y, pz, px, py
#ifdef AKREON_DEBUG
                                  ,
                                  trace_state
#endif
          );
      if (linalg::is_fatal_event(next_event))
        return std::tuple(std::move(new_pz_rhs), next_event);

      if constexpr (idx + 1 == NumCones) {
        const auto [pz_next, solve_event] = S_inv.solve(new_pz_rhs + rd);
        if (linalg::is_fatal_event(solve_event))
          return std::tuple(std::move(new_pz_rhs), solve_event);
        pz = pz_next;
      }

      const auto px_local = F_blk * pz + rpk;
      const auto py_local = T(-1) * rhs_c - Hes.apply(px_local);

      for (size_t i = Start, j = 0; i < End; ++i, ++j) {
        px(i) = px_local(j), py(i) = py_local(j);
      }
#ifdef AKREON_DEBUG
      for (size_t j = 0; j < N; ++j) cone_trace.pz[j] = pz(j);
      for (size_t j = 0; j < End - Start; ++j) {
        cone_trace.px[j] = px_local(j);
        cone_trace.py[j] = py_local(j);
      }
#endif

      return std::tuple(std::move(new_pz_rhs), linalg::NumericalEvent::NONE);
    }
  }
};

}  // namespace convex
