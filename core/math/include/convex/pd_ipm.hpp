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

#include <iostream>

#include "convex_problem.hpp"
#include "linalg/operator/schur_kkt.hpp"

namespace convex {

/**
 * @brief mu - normalized duality gap
 *
 * <x, s> / nu
 *
 * @param x \in K
 * @param s \in K^*
 * @param nu \sum of barrier param across cones
 *
 * @return mu
 */
template <linalg::VectorLike ConeVar>
constexpr auto mu(const ConeVar& x, const ConeVar& y,
                  const linalg::vec_type_t<ConeVar>& nu) {
  return linalg::dot(x, y) / nu;
}

/**
 * @brief PrimalDualResult
 */
template <linalg::VectorLike Var, linalg::VectorLike ConeVar>
struct PrimalDualResult {
  using T = linalg::vec_type_t<Var>;

  linalg::SolverResult tag{};  ///< tag for result of pd-ipm
  T mu{};                      ///< normalized duality gap
  Var rd{};                    ///< dual residuals
  ConeVar rp{};                ///< primal residuals
  Var z{};                     ///< decision var \in Rn + aux lifts
  ConeVar x{};                 ///< primal cone var \in K
  ConeVar y{};                 ///< dual cone var \in K^*
};

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
  Var predictor(Var&& pz_rhs, const auto& K, const auto& S_inv, const Mat& F,
                const ConeVar& rp, const Var& rd, const ConeVar& x,
                const ConeVar& y, Var& pz, ConeVar& px, ConeVar& py) const {
    if constexpr (idx >= NumCones)
      return pz_rhs;
    else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;
      constexpr size_t KiRows = linalg::vec_size_v<Slack>;
      constexpr size_t End = Start + KiRows;

      if (conic::is_zoc_v<decltype(Ki)>)
        return predictor<idx + 1, End>(std::move(pz_rhs), K, S_inv, F, rp, rd,
                                       x, y, pz, px, py);

      using ConePoint = linalg::Vector<T, End - Start>;
      ConePoint xk{}, yk{}, pxk{}, pyk{}, rpk{};

      for (size_t i = Start, j = 0; i < End; ++i, ++j) {
        xk(j) = x(i);
        yk(j) = y(i);
        pxk(j) = px(i);
        pyk(j) = py(i);
        rpk(j) = rp(i);
      }

      const auto Hes = Ki.H(xk, yk);
      const auto F_blk = linalg::get_block<Start, End, 0, N>(F);
      const auto F_blk_T = linalg::transpose(F_blk);

      ConePoint rhs_c = T(-1) * Ki.circ(xk, yk);

      pz_rhs = pz_rhs - F_blk_T * Hes.apply(rpk) + F_blk_T * rhs_c;

      const auto new_pz_rhs = predictor<idx + 1, End>(
          std::move(pz_rhs), K, S_inv, F, rp, rd, x, y, pz, px, py);

      if constexpr (idx + 1 == NumCones) pz = S_inv.solve(new_pz_rhs + rd);

      const auto px_tmp = F_blk * pz + rpk;

      const auto py_tmp = rhs_c - Hes.apply(px_tmp);

      for (size_t i = Start, j = 0; i < End; ++i, ++j) {
        px(i) = px_tmp(j), py(i) = py_tmp(j);
      }

      return new_pz_rhs;
    }
  }

  /**
   * @brief corrector (centering) direction
   *
   * @param pz_rhs (forward pass accumulate)
   * @param mu_tgt (mu target for mehrotra centering)
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
  Var corrector(Var&& pz_rhs, const T& mu_tgt, const ConeVar& x_pred,
                const ConeVar& y_pred, const auto& K, const auto& S_inv,
                const Mat& F, const ConeVar& rp, const Var& rd,
                const ConeVar& x, const ConeVar& y, Var& pz, ConeVar& px,
                ConeVar& py) const {
    if constexpr (idx >= NumCones)
      return pz_rhs;
    else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;
      constexpr size_t KiRows = linalg::vec_size_v<Slack>;
      constexpr size_t End = Start + KiRows;

      if (conic::is_zoc_v<decltype(Ki)>)
        return corrector<idx + 1, End>(std::move(pz_rhs), mu_tgt, x_pred,
                                       y_pred, K, S_inv, F, rp, rd, x, y, pz,
                                       px, py);

      using ConePoint = linalg::Vector<T, End - Start>;
      ConePoint xk{}, yk{}, pxk{}, pyk{}, rpk{}, x_pred_k{}, y_pred_k{};

      for (size_t i = Start, j = 0; i < End; ++i, ++j) {
        xk(j) = x(i);
        yk(j) = y(i);
        pxk(j) = px(i);
        pyk(j) = py(i);
        x_pred_k(j) = x_pred(i);
        y_pred_k(j) = y_pred(i);
        rpk(j) = rp(i);
      }

      const auto Hes = Ki.H(xk, yk);
      const auto F_blk = linalg::get_block<Start, End, 0, N>(F);
      const auto F_blk_T = linalg::transpose(F_blk);

      ConePoint rhs_c =
          mu_tgt * Ki.e() - Ki.circ(x_pred_k, y_pred_k) - Ki.circ(xk, yk);

      pz_rhs = pz_rhs - F_blk_T * Hes.apply(rpk) + F_blk_T * rhs_c;

      const auto new_pz_rhs =
          corrector<idx + 1, End>(std::move(pz_rhs), mu_tgt, x_pred, y_pred, K,
                                  S_inv, F, rp, rd, x, y, pz, px, py);

      if constexpr (idx + 1 == NumCones) pz = S_inv.solve(new_pz_rhs + rd);

      const auto px_tmp = F_blk * pz + rpk;

      const auto py_tmp = rhs_c - Hes.apply(px_tmp);

      for (size_t i = Start, j = 0; i < End; ++i, ++j) {
        px(i) = px_tmp(j), py(i) = py_tmp(j);
      }

      return new_pz_rhs;
    }
  }
};

/**
 * @brief Primal Dual Interior Point Method Solver
 */
template <ProblemLike Problem>
struct PrimalDualIPM {
  /// @brief alias and helpers to get types & sizes
  using Builder = prob_builder_t<Problem>;
  using DecVar = builder_dom_t<Builder>;
  using T = linalg::space_type_t<DecVar>;
  static constexpr size_t NumCones =
      conic::cone_block_num_cones_v<builder_cb_t<Builder>>;

  /**
   * @brief solve
   *
   * @param prob ConvexProblem
   *
   * @return result
   */
  auto solve(const ConvexProblem<Builder, StageType::PROBLEM_SET>& prob) const {
    const auto c = linalg::as_vector(prob.linear_objective());
    const auto g = prob.constraint_map().translation();
    const auto F = prob.constraint_map().linear_part();
    const auto K = prob.constraint_cones();

    static constexpr size_t M = linalg::mat_rows_v<decltype(F)>;
    static constexpr size_t N = linalg::mat_cols_v<decltype(F)>;
    static_assert(N == linalg::mat_cols_v<decltype(F)>,
                  "Expected decision var dim != N");

    using Schur = linalg::Matrix<T, N, N>;
    using Var = linalg::Vector<T, N>;
    using ConeVar = linalg::Vector<T, M>;

    std::cout << "c: ";
    std::for_each(c.begin(), c.end(),
                  [&](const auto& elem) { std::cout << elem << " "; });
    std::cout << std::endl;

    std::cout << "F: ";
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        std::cout << F(i)(j) << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "g: ";
    std::for_each(g.begin(), g.end(),
                  [&](const auto& elem) { std::cout << elem << " "; });
    std::cout << std::endl;

    PrimalDualResult<Var, ConeVar> res;
    if (!prob.validate()) return res;

    auto dual_residual = [&F, &c](const auto& y) {
      return linalg::transpose(F) * y - c;
    };

    auto primal_residual = [&F, &g](const auto& z, const auto& x) {
      return F * z + g - x;
    };

    auto get_tag = [](const auto& primal_feas, const auto& dual_feas,
                      const auto& gap_min) {
      linalg::SolverResult tag;
      if (primal_feas && dual_feas && gap_min)
        tag = linalg::SolverResult::OPTIMAL;
      else if (primal_feas && dual_feas)
        tag = linalg::SolverResult::PRIMAL_DUAL_FEASIBLE;
      else if (primal_feas)
        tag = linalg::SolverResult::PRIMAL_FEASIBLE;
      else if (dual_feas)
        tag = linalg::SolverResult::DUAL_FEASIBLE;
      else
        tag = linalg::SolverResult::INFEASIBLE_OR_UNBOUNDED;
      return tag;
    };

    auto& z = res.z;
    auto& x = res.x;
    auto& y = res.y;
    auto& rp = res.rp;
    auto& rd = res.rd;
    auto& tag = res.tag;

    PredictorCorrector<NumCones, decltype(F), Var, ConeVar> pc;

    seed_cone_vars(K, x, y);

    for (size_t i = 0; i < linalg::SolverTraits<T>::STEPS &&
                       tag != linalg::SolverResult::OPTIMAL;
         ++i) {
      std::cout << "iter " << i;

      ConeVar px_pred{}, px_corr{}, py_pred{}, py_corr{};
      Var pz_pred{}, pz_corr{};

      /// (dakre) 1 - accumulate and factor schur & nu
      const auto params = accumulate_params(Schur{}, T(0), F, K, x, y);
      const auto S = std::get<0>(params);
      const auto nu = std::get<1>(params);
      std::cout << ": nu " << nu;

      const auto ldl = linalg::LDL(S);

      /// (dakre) 2 - compute initial primal & dual residuals (negated)
      rd = dual_residual(y);
      rp = primal_residual(z, x);

      std::cout << ", ||rd|| " << linalg::L2<decltype(rd)>()(rd);
      std::cout << ", ||rp|| " << linalg::L2<decltype(rp)>()(rp);

      /// (dakre) 3 - predictor step
      pc.predictor(Var{}, K, ldl, F, rp, rd, x, y, pz_pred, px_pred, py_pred);

      /// (dakre) 4 - mehrotra centering
      auto alpha_xy = alpha_max(T(1), T(1), K, x, y, px_pred, py_pred);
      auto alpha = std::min(std::get<0>(alpha_xy), std::get<1>(alpha_xy));
      std::cout << ", alpha_x " << std::get<0>(alpha_xy);
      std::cout << ", alpha_y " << std::get<1>(alpha_xy);
      const auto x_aff = x + std::get<0>(alpha_xy) * px_pred;
      const auto y_aff = y + std::get<1>(alpha_xy) * py_pred;
      const auto mu_aff = mu(x_aff, y_aff, nu);
      std::cout << " mu_aff " << mu_aff;
      auto sigma = linalg::constexpr_pow(mu_aff / mu(x, y, nu), 3);
      sigma = std::max(T(0), std::min(T(1.0), sigma));

      std::cout << ", sigma " << sigma;
      const auto mu_tgt = sigma * mu(x, y, nu);
      std::cout << ", mu_tgt " << mu_tgt;

      /// (dakre) 5 - corrector step
      pc.corrector(Var{}, mu_tgt, x_aff, y_aff, K, ldl, F, rp, rd, x, y,
                   pz_corr, px_corr, py_corr);

      /// (dakre) 6 - linear combination between predictor & corrector
      /// dirs to form uniform newton step
      const auto px = px_pred + px_corr;
      const auto py = py_pred + py_corr;
      const auto pz = pz_pred + pz_corr;

      /// (dakre) 7 - alpha max
      alpha_xy = alpha_max(T(1), T(1), K, x, y, px, py);
      alpha = std::min(std::get<0>(alpha_xy), std::get<1>(alpha_xy));
      std::cout << ", alpha " << alpha;

      /// (dakre) 8 - take newton step scaled by alpha
      x = x + alpha * px;
      y = y + alpha * py;
      z = z + alpha * pz;

      /// (dakre) 9 - compute residual again after update
      rd = dual_residual(y);
      rp = primal_residual(z, x);

      std::cout << ", ||rd|| " << linalg::L2<decltype(rd)>()(rd);
      std::cout << ", ||rp|| " << linalg::L2<decltype(rp)>()(rp);
      std::cout << ", mu(x,y,nu) " << mu(x, y, nu) << std::endl;

      bool dual_feas = linalg::L2<decltype(rd)>()(rd) <=
                       linalg::SolverTraits<T>::DUAL_FEASIBLE;
      bool primal_feas = linalg::L2<decltype(rp)>()(rp) <=
                         linalg::SolverTraits<T>::PRIMAL_FEASIBLE;
      res.mu = mu(x, y, nu);
      bool gap_min = res.mu <= linalg::SolverTraits<T>::DUALITY_GAP;

      tag = get_tag(primal_feas, dual_feas, gap_min);
    }

    std::cout << "z ";
    std::for_each(res.z.begin(), res.z.end(),
                  [&](const auto& elem) { std::cout << elem << " "; });
    std::cout << std::endl;

    std::cout << "x ";
    std::for_each(res.x.begin(), res.x.end(),
                  [&](const auto& elem) { std::cout << elem << " "; });
    std::cout << std::endl;

    std::cout << "y ";
    std::for_each(res.y.begin(), res.y.end(),
                  [&](const auto& elem) { std::cout << elem << " "; });
    std::cout << std::endl;

    return res;
  }

 private:
  /**
   * @brief accumulate params (Schur & nu)
   *
   * \tilde{KKT} = [ 0 F^T 0   ] [ pz; py; px ] = -[ rd; rp; rc ]
   *               [ F 0   -I  ]
   *               [ 0 I   H_x ]
   *
   * S = F^T H_x F where S += S_i for each cone
   *
   * @param S Schur matrix
   * @param nu \sum of barrier params across K
   * @param F mapping from R^n -> conic space
   * @param K cartesian product of cones
   * @param x \in K
   * @param y \in K^*
   *
   * @return (S, nu)
   */
  template <size_t idx = 0, size_t Start = 0, linalg::MatrixLike Schur,
            linalg::MatrixLike Mat, linalg::VectorLike ConeVar>
  constexpr auto accumulate_params(Schur&& S, T&& nu, const Mat& F,
                                   const auto& K, const ConeVar& x,
                                   const ConeVar& y) const {
    if constexpr (idx >= NumCones)
      return std::tuple(S, nu);
    else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;

      constexpr size_t N = linalg::mat_cols_v<decltype(F)>;
      constexpr size_t KiRows = linalg::vec_size_v<Slack>;
      constexpr size_t End = Start + KiRows;

      if (conic::is_zoc_v<decltype(Ki)>)
        return accumulate_params<idx + 1, End>(std::move(S), std::move(nu), F,
                                               K, x, y);

      using ConePoint = linalg::Vector<T, End - Start>;
      ConePoint xk{}, yk{};

      for (size_t i = Start, j = 0; i < End; ++i, ++j)
        xk(j) = x(i), yk(j) = y(i);

      const auto Hes = Ki.H(xk, yk);

      const auto F_blk = get_block<Start, End, 0, N>(F);
      const linalg::SchurKKT<decltype(Hes.H), decltype(F_blk)> kkt;
      const auto Si = kkt.make(Hes, F_blk);
      S = S + Si;

      nu += Ki.nu();

      return accumulate_params<idx + 1, End>(std::move(S), std::move(nu), F, K,
                                             x, y);
    }
  }

  /**
   * @brief seed cone vars
   *
   * Initialize x \in K and y \in K^* with initial primal & dual points in
   * each cone
   *
   * @param K cartesian product of cones
   * @param x \in K
   * @param y \in K^*
   *
   * @return (void)
   */
  template <size_t idx = 0, size_t Start = 0, linalg::VectorLike ConeVar>
  constexpr void seed_cone_vars(const auto& K, ConeVar& x, ConeVar& y) const {
    if constexpr (idx >= NumCones)
      return;
    else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;

      constexpr size_t KiRows = linalg::vec_size_v<Slack>;
      constexpr size_t End = Start + KiRows;

      if (conic::is_zoc_v<decltype(Ki)>)
        return seed_cone_vars<idx + 1, End>(K, x, y);

      const auto x0 = Ki.x0(), s0 = Ki.s0();
      for (size_t i = Start, j = 0; i < End; ++i, ++j)
        x(i) = x0(j), y(i) = s0(j);

      seed_cone_vars<idx + 1, End>(K, x, y);
    }
  }

  /**
   * @brief alpha max
   *
   * for each cone compute alpha max and return
   *
   * @param alpha_x max step to boundary
   * @param alpha_y max step to boundary
   * @param K cartesian product of cones
   * @param x \in K
   * @param y \in K^*
   * @param px newton step in x dir
   * @param py newton step in y dir
   *
   * @return \alpha_max
   */
  template <size_t idx = 0, size_t Start = 0, linalg::VectorLike ConeVar>
  constexpr auto alpha_max(T&& alpha_x, T&& alpha_y, const auto& K,
                           const ConeVar& x, const ConeVar& y,
                           const ConeVar& px, const ConeVar& py) const {
    if constexpr (idx >= NumCones)
      return std::tuple(alpha_x, alpha_y);
    else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;
      constexpr size_t KiRows = linalg::vec_size_v<Slack>;
      constexpr size_t End = Start + KiRows;

      if (conic::is_zoc_v<decltype(Ki)>)
        return alpha_max<idx + 1, End>(std::move(alpha_x), std::move(alpha_y),
                                       K, x, y, px, py);

      using ConePoint = linalg::Vector<T, End - Start>;
      ConePoint xk{}, yk{}, pxk{}, pyk{};

      for (size_t i = Start, j = 0; i < End; ++i, ++j)
        xk(j) = x(i), yk(j) = y(i), pxk(j) = px(i), pyk(j) = py(i);

      alpha_x = std::min(Ki.alpha_max(xk, pxk), alpha_x);
      alpha_y = std::min(Ki.alpha_max(yk, pyk), alpha_y);

      return alpha_max<idx + 1, End>(std::move(alpha_x), std::move(alpha_y), K,
                                     x, y, px, py);
    }
  }
};

}  // namespace convex
