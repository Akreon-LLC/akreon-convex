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

#include "convex/cone_traits.hpp"
#include "convex/convex_traits.hpp"
#include "equality_coupling.hpp"
#include "linalg/linalg_util.hpp"
#include "newton_step.hpp"
#ifdef AKREON_REPORT_OUTPUT
#include "pd_ipm_output.hpp"
#endif

namespace convex {

/**
 * @brief Predictor Corrector
 */
template <size_t NumCones, linalg::MatrixLike ConeMat, linalg::MatrixLike EqMat,
          linalg::VectorLike Var, linalg::VectorLike ConeVar>
struct PredictorCorrector {
  /// @brief extract T and N from Mat
  using T = linalg::mat_type_t<ConeMat>;
  static constexpr size_t N = linalg::mat_cols_v<ConeMat>;

  static constexpr size_t EqRows = linalg::mat_rows_v<EqMat>;
  using NuVar = linalg::Vector<T, EqRows>;
  using Step = NewtonStep<Var, ConeVar, NuVar>;

  /**
   * @brief predictor (affine-scaling) direction
   *
   * @param pz_rhs (rhs forward pass accumulate)
   * @param K (cartesian product of cones)
   * @param M_inv (factored M = (F^T H F)^{-1})
   * @param eq (equality coupling)
   * @param F (R^n -> conic space)
   * @param rp (primal residual)
   * @param rd (dual residual)
   * @param req (equality residual)
   * @param x (x \in K iterate)
   * @param y (y \in K^* iterate)
   *
   * @return pz_rhs, step (ignore rhs on return), event
   */
  template <size_t idx = 0, size_t Start = 0>
  constexpr auto predictor(Var&& pz_rhs, const auto& K, const auto& M_inv,
                           const auto& eq, const ConeMat& F, const ConeVar& rp,
                           const Var& rd, const NuVar& req, const ConeVar& x,
                           const ConeVar& y
#ifdef AKREON_REPORT_OUTPUT
                           ,
                           IterState<T, NumCones, linalg::mat_rows_v<ConeMat>,
                                     N, EqRows>& trace_state
#endif
  ) const {
    if constexpr (idx >= NumCones)
      return std::tuple(std::move(pz_rhs), Step{},
                        linalg::NumericalEvent::NONE);
    else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;
      constexpr size_t KiRows = linalg::vec_size_v<Slack>;
      constexpr size_t End = Start + KiRows;

      using ConePoint = linalg::Vector<T, End - Start>;
      ConePoint xk{}, yk{}, rpk{};

      for (size_t i = Start, j = 0; i < End; ++i, ++j) {
        xk(j) = x(i);
        yk(j) = y(i);
        rpk(j) = rp(i);
      }

      const auto [P, event] = Ki.H(xk, yk);
      if (linalg::is_fatal_event(event))
        return std::tuple(std::move(pz_rhs), Step{}, event);

      const auto F_blk = linalg::get_block<Start, End, 0, N>(F);
      const auto F_blk_T = linalg::transpose(F_blk);

      ConePoint rhs_c = yk;

      pz_rhs = pz_rhs - F_blk_T * P.apply(rpk) - F_blk_T * rhs_c;
#ifdef AKREON_REPORT_OUTPUT
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

      auto [new_pz_rhs, step, next_event] = predictor<idx + 1, End>(
          std::move(pz_rhs), K, M_inv, eq, F, rp, rd, req, x, y
#ifdef AKREON_REPORT_OUTPUT
          ,
          trace_state
#endif
      );
      if (linalg::is_fatal_event(next_event))
        return std::tuple(std::move(new_pz_rhs), std::move(step), next_event);

      if constexpr (idx + 1 == NumCones) {
        const auto [pz_next, pnu_next, solve_event] =
            eq.solve_step(M_inv, new_pz_rhs, rd, req);
        if (linalg::is_fatal_event(solve_event))
          return std::tuple(std::move(new_pz_rhs), std::move(step),
                            solve_event);
        step.pz = pz_next;
        step.pnu = pnu_next;
      }

      const auto px_local = F_blk * step.pz + rpk;
      const auto py_local = T(-1) * rhs_c - P.apply(px_local);

      for (size_t i = Start, j = 0; i < End; ++i, ++j) {
        step.px(i) = px_local(j), step.py(i) = py_local(j);
      }
#ifdef AKREON_REPORT_OUTPUT
      for (size_t j = 0; j < N; ++j) cone_trace.pz[j] = step.pz(j);
      for (size_t j = 0; j < EqRows; ++j) cone_trace.pnu[j] = step.pnu(j);
      for (size_t j = 0; j < End - Start; ++j) {
        cone_trace.px[j] = px_local(j);
        cone_trace.py[j] = py_local(j);
      }
#endif

      return std::tuple(std::move(new_pz_rhs), std::move(step),
                        linalg::NumericalEvent::NONE);
    }
  }

  /**
   * @brief corrector (centering) direction
   *
   * @param pz_rhs (forward pass accumulate)
   * @param px_pred (newton step affine scaling dir x)
   * @param py_pred (newton step affine scaling dir y)
   * @param K (cartesian product of cones)
   * @param M_inv (factored M = (F^T H F)^{-1})
   * @param eq (equality coupling)
   * @param F (R^n -> conic space)
   * @param rp (primal residual)
   * @param rd (dual residual)
   * @param req (equality residual)
   * @param x (x \in K iterate)
   * @param y (y \in K^* iterate)
   * @return pz_rhs, step (ignore rhs on return), event
   */
  template <size_t idx = 0, size_t Start = 0>
  constexpr auto corrector(Var&& pz_rhs, const ConeVar& px_aff,
                           const ConeVar& py_aff, const auto& K,
                           const auto& M_inv, const auto& eq, const ConeMat& F,
                           const ConeVar& rp, const Var& rd, const NuVar& req,
                           const ConeVar& x, const ConeVar& y
#ifdef AKREON_REPORT_OUTPUT
                           ,
                           IterState<T, NumCones, linalg::mat_rows_v<ConeMat>,
                                     N, EqRows>& trace_state
#endif
  ) const {
    if constexpr (idx >= NumCones)
      return std::tuple(std::move(pz_rhs), Step{},
                        linalg::NumericalEvent::NONE);
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

      const auto [P, event] = Ki.H(xk, yk);
      if (linalg::is_fatal_event(event))
        return std::tuple(std::move(pz_rhs), Step{}, event);

      const auto F_blk = linalg::get_block<Start, End, 0, N>(F);
      const auto F_blk_T = linalg::transpose(F_blk);

      T mu_k = Ki.trace(xk, yk) / Ki.nu();
      const auto [grad_xk, grad_event] = Ki.grad(xk);
      if (linalg::is_fatal_event(grad_event))
        return std::tuple(std::move(pz_rhs), Step{}, grad_event);

      ConePoint rhs_c = yk + mu_k * grad_xk + Ki.circ(px_aff_k, py_aff_k);

      pz_rhs = pz_rhs - F_blk_T * P.apply(rpk) - F_blk_T * rhs_c;
#ifdef AKREON_REPORT_OUTPUT
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

      auto [new_pz_rhs, step, next_event] = corrector<idx + 1, End>(
          std::move(pz_rhs), px_aff, py_aff, K, M_inv, eq, F, rp, rd, req, x, y
#ifdef AKREON_REPORT_OUTPUT
          ,
          trace_state
#endif
      );
      if (linalg::is_fatal_event(next_event))
        return std::tuple(std::move(new_pz_rhs), std::move(step), next_event);

      if constexpr (idx + 1 == NumCones) {
        const auto [pz_next, pnu_next, solve_event] =
            eq.solve_step(M_inv, new_pz_rhs, rd, req);
        if (linalg::is_fatal_event(solve_event))
          return std::tuple(std::move(new_pz_rhs), std::move(step),
                            solve_event);
        step.pz = pz_next;
        step.pnu = pnu_next;
      }

      const auto px_local = F_blk * step.pz + rpk;
      const auto py_local = T(-1) * rhs_c - P.apply(px_local);

      for (size_t i = Start, j = 0; i < End; ++i, ++j) {
        step.px(i) = px_local(j), step.py(i) = py_local(j);
      }
#ifdef AKREON_REPORT_OUTPUT
      for (size_t j = 0; j < N; ++j) cone_trace.pz[j] = step.pz(j);
      for (size_t j = 0; j < EqRows; ++j) cone_trace.pnu[j] = step.pnu(j);
      for (size_t j = 0; j < End - Start; ++j) {
        cone_trace.px[j] = px_local(j);
        cone_trace.py[j] = py_local(j);
      }
#endif

      return std::tuple(std::move(new_pz_rhs), std::move(step),
                        linalg::NumericalEvent::NONE);
    }
  }
};

} // namespace convex
