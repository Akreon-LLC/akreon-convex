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
  size_t num_iter{};           ///< total number of iter
};

#ifdef AKREON_DEBUG
#include <array>
#include <cstddef>
#include <cstdio>

/**
 * @brief IterState
 */
template <linalg::FieldLike T>
struct IterState {
  T mu{};       ///< normalized duality gap
  T mu_aff{};   ///< duality gap of affine-scaling step
  T sigma{};    ///< mehrotra centering correction
  T mu_tgt{};   ///< mehrotra mu target for centering
  T alpha_x{};  ///< max step to dK size
  T alpha_y{};  ///< max step to dK^* size
  T rd_norm{};  ///< dual norm residual size
  T rp_norm{};  ///< primal norm residual size
  T pz_norm{};  ///< newton step in z \in R^n size
  T px_norm{};  ///< newton step in x \in K size
  T py_norm{};  /// <newton step in y \in K^* size

  /**
   * @flags
   *
   * first bit : primal feasible
   * second bit : dual feasible
   * third bit : dual gap minimized
   */
  uint8_t flags{};  ///< status of iter
};

/**
 * @brief iter state to c string
 *
 * @param iter state,
 * @param char buffer
 * @param buffer size
 *
 * @return write position
 */
template <linalg::FieldLike T>
inline size_t iter_state_to_cstr(const IterState<T>& state, char* out,
                                 const size_t out_size) {
  AK_ASSERT(out != nullptr && "iter_state_to_cstr requires non-null output");
  if (out == nullptr || out_size == 0) return 0;

  const int bytes = std::snprintf(
      out, out_size,
      "mu=%g mu_aff=%g sigma=%g mu_tgt=%g ax=%g ay=%g rd=%g rp=%g pz=%g px=%g "
      "py=%g flags=0x%02X",
      static_cast<double>(state.mu), static_cast<double>(state.mu_aff),
      static_cast<double>(state.sigma), static_cast<double>(state.mu_tgt),
      static_cast<double>(state.alpha_x), static_cast<double>(state.alpha_y),
      static_cast<double>(state.rd_norm), static_cast<double>(state.rp_norm),
      static_cast<double>(state.pz_norm), static_cast<double>(state.px_norm),
      static_cast<double>(state.py_norm), state.flags);

  if (bytes < 0) {
    out[0] = '\0';
    return 0;
  }

  if (static_cast<size_t>(bytes) >= out_size) {
    out[out_size - 1] = '\0';
    return out_size - 1;
  }

  return static_cast<size_t>(bytes);
}

/**
 * @brief trace to cstring
 *
 * @param number of iters
 * @param trace (array of iter states)
 * @param buffer char string
 * @param buffer size
 *
 * @return write position (in case buffer to small)
 */
template <linalg::FieldLike T, size_t N>
inline size_t trace_to_cstr(const size_t& num_iter,
                            const std::array<IterState<T>, N>& trace, char* out,
                            const size_t& out_size) {
  AK_ASSERT(out != nullptr && "trace_to_cstr requires non-null output");
  if (out == nullptr || out_size == 0) return 0;

  const size_t used_iters = (num_iter < N) ? num_iter : N;
  size_t write_pos{0};
  for (size_t i = 0; i < used_iters; ++i) {
    const int header =
        std::snprintf(out + write_pos, out_size - write_pos, "[%zu] ", i);
    if (header < 0) break;
    if (static_cast<size_t>(header) >= out_size - write_pos) {
      out[out_size - 1] = '\0';
      return out_size - 1;
    }
    write_pos += static_cast<size_t>(header);

    const size_t body =
        iter_state_to_cstr(trace[i], out + write_pos, out_size - write_pos);
    write_pos += body;
    if (write_pos >= out_size - 1) {
      out[out_size - 1] = '\0';
      return out_size - 1;
    }

    out[write_pos++] = '\n';
    out[write_pos] = '\0';
  }
  return write_pos;
}
#endif

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
  constexpr Var predictor(Var&& pz_rhs, const auto& K, const auto& S_inv,
                          const Mat& F, const ConeVar& rp, const Var& rd,
                          const ConeVar& x, const ConeVar& y, Var& pz,
                          ConeVar& px, ConeVar& py) const {
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

      ConePoint rhs_c = Ki.circ(xk, yk);

      pz_rhs = pz_rhs - F_blk_T * Hes.apply(rpk) - F_blk_T * rhs_c;

      const auto new_pz_rhs = predictor<idx + 1, End>(
          std::move(pz_rhs), K, S_inv, F, rp, rd, x, y, pz, px, py);

      if constexpr (idx + 1 == NumCones) pz = S_inv.solve(new_pz_rhs + rd);

      const auto px_tmp = F_blk * pz + rpk;

      const auto py_tmp = T(-1) * rhs_c - Hes.apply(px_tmp);

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
  constexpr Var corrector(Var&& pz_rhs, const T& mu_tgt, const T& aff_scale_sq,
                          const ConeVar& dx_aff, const ConeVar& dy_aff,
                          const auto& K, const auto& S_inv, const Mat& F,
                          const ConeVar& rp, const Var& rd, const ConeVar& x,
                          const ConeVar& y, Var& pz, ConeVar& px,
                          ConeVar& py) const {
    if constexpr (idx >= NumCones)
      return pz_rhs;
    else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;
      constexpr size_t KiRows = linalg::vec_size_v<Slack>;
      constexpr size_t End = Start + KiRows;

      if (conic::is_zoc_v<decltype(Ki)>)
        return corrector<idx + 1, End>(std::move(pz_rhs), mu_tgt, aff_scale_sq,
                                       dx_aff, dy_aff, K, S_inv, F, rp, rd, x,
                                       y, pz, px, py);

      using ConePoint = linalg::Vector<T, End - Start>;
      ConePoint xk{}, yk{}, dx_aff_k{}, dy_aff_k{}, rpk{};

      for (size_t i = Start, j = 0; i < End; ++i, ++j) {
        xk(j) = x(i);
        yk(j) = y(i);
        dx_aff_k(j) = dx_aff(i);
        dy_aff_k(j) = dy_aff(i);
        rpk(j) = rp(i);
      }

      const auto Hes = Ki.H(xk, yk);
      const auto F_blk = linalg::get_block<Start, End, 0, N>(F);
      const auto F_blk_T = linalg::transpose(F_blk);

      ConePoint rhs_c =
          aff_scale_sq * Ki.circ(dx_aff_k, dy_aff_k) - mu_tgt * Ki.e();

      pz_rhs = pz_rhs - F_blk_T * Hes.apply(rpk) - F_blk_T * rhs_c;

      const auto new_pz_rhs = corrector<idx + 1, End>(
          std::move(pz_rhs), mu_tgt, aff_scale_sq, dx_aff, dy_aff, K, S_inv, F,
          rp, rd, x, y, pz, px, py);

      if constexpr (idx + 1 == NumCones) pz = S_inv.solve(new_pz_rhs + rd);

      const auto px_tmp = F_blk * pz + rpk;

      const auto py_tmp = T(-1) * rhs_c - Hes.apply(px_tmp);

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
  /// @brief alias and helpers to get types
  using Builder = prob_builder_t<Problem>;
  using DecVar = conic::builder_dom_t<Builder>;
  using CB = conic::builder_cb_t<Builder>;
  using FMap = conic::cone_block_map_t<CB>;
  using FMat = affine::affine_mat_t<FMap>;
  using T = linalg::space_type_t<DecVar>;

  /// @brief helpers for size of cones & F mat
  static constexpr size_t NumCones =
      conic::cone_block_num_cones_v<conic::builder_cb_t<Builder>>;
  static constexpr size_t M = linalg::mat_rows_v<FMat>;
  static constexpr size_t N = linalg::mat_cols_v<FMat>;

  /// @brief useful types for PD-IPM
  using Schur = linalg::Matrix<T, N, N>;
  using Var = linalg::Vector<T, N>;
  using ConeVar = linalg::Vector<T, M>;

  static_assert(is_problem_set_v<Problem>,
                "Problem is not set therefore cannot be solved");

  /**
   * @CTOR
   *
   * @param prob : ConvexProblem
   */
  constexpr explicit PrimalDualIPM(const Problem& prob) { solve(prob); }

  /** @brief member vars */
  PrimalDualResult<Var, ConeVar> res;  ///< PD-IPM result
#ifdef AKREON_DEBUG
  std::array<IterState<T>, linalg::SolverTraits<T>::PD_IPM_STEPS>
      trace;  ///< debug iteration trace
#endif

 private:
  /**
   * @brief solve
   *
   * @param prob ConvexProblem
   *
   * @return result
   */
  constexpr void solve(const Problem& prob) {
    const auto c = linalg::as_vector(prob.linear_objective());
    const auto g = prob.constraint_map().translation();
    const auto F = prob.constraint_map().linear_part();
    const auto K = prob.constraint_cones();

    if (!prob.validate()) return;

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
    auto& i = res.num_iter;

    PredictorCorrector<NumCones, decltype(F), Var, ConeVar> pc;

    seed_cone_vars(K, x, y);

    bool dual_feas{false}, primal_feas{false}, gap_min{false};
    for (i = 0; i < linalg::SolverTraits<T>::PD_IPM_STEPS &&
                tag != linalg::SolverResult::OPTIMAL;
         ++i) {
      ConeVar px_pred{}, px_corr{}, py_pred{}, py_corr{};
      Var pz_pred{}, pz_corr{};

      /// (dakre) 1 - accumulate and factor schur & nu
      const auto params = accumulate_params(Schur{}, T(0), F, K, x, y);
      const auto S = std::get<0>(params);
      const auto nu = std::get<1>(params);

      const auto ldl = linalg::LDL(S);

      /// (dakre) 2 - compute initial primal & dual residuals (negated)
      rd = dual_residual(y);
      rp = primal_residual(z, x);
      auto rd_norm = linalg::L2<decltype(rd)>()(rd);
      auto rp_norm = linalg::L2<decltype(rp)>()(rp);

      /// (dakre) 3 - predictor step
      pc.predictor(Var{}, K, ldl, F, rp, rd, x, y, pz_pred, px_pred, py_pred);

      /// (dakre) 4 - mehrotra centering
      auto alpha_xy = alpha_max(T(1), T(1), K, x, y, px_pred, py_pred);
      const auto x_aff = x + std::get<0>(alpha_xy) * px_pred;
      const auto y_aff = y + std::get<1>(alpha_xy) * py_pred;
      const auto mu_curr = mu(x, y, nu);
      const auto mu_aff = mu(x_aff, y_aff, nu);

      auto sigma = linalg::constexpr_pow(mu_aff / mu_curr, 3);
      sigma = std::max(T(0), std::min(T(1.0), sigma));
      T theta = T(0.7);
      if (primal_feas && dual_feas) theta = T(0.3), sigma *= theta;

      const auto mu_tgt = sigma * mu_curr;
      const auto aff_scale_sq = std::get<0>(alpha_xy) * std::get<1>(alpha_xy);

      /// (dakre) 5 - corrector step
      pc.corrector(Var{}, mu_tgt, aff_scale_sq, px_pred, py_pred, K, ldl, F,
                   ConeVar{}, Var{}, x, y, pz_corr, px_corr, py_corr);

      /// (dakre) 6 - blend predictor/corrector to avoid over-aggressive
      /// second-order correction on near-feasible iterates.
      const auto px = px_pred + theta * px_corr;
      const auto py = py_pred + theta * py_corr;
      const auto pz = pz_pred + theta * pz_corr;

      /// (dakre) 7 - alpha max
      alpha_xy = alpha_max(T(1), T(1), K, x, y, px, py);
      auto alpha_y = std::get<1>(alpha_xy);
      if (primal_feas && dual_feas) alpha_y *= theta;

#ifdef AKREON_DEBUG
      trace[i].mu = res.mu;
      trace[i].mu_aff = mu_aff;
      trace[i].sigma = sigma;
      trace[i].mu_tgt = mu_tgt;
      trace[i].alpha_x = std::get<0>(alpha_xy);
      trace[i].alpha_y = std::get<1>(alpha_xy);
      trace[i].rd_norm = rd_norm;
      trace[i].rp_norm = rp_norm;
      trace[i].pz_norm = linalg::L2<decltype(pz)>()(pz);
      trace[i].px_norm = linalg::L2<decltype(px)>()(px);
      trace[i].py_norm = linalg::L2<decltype(py)>()(py);
#endif

      /// (dakre) 8 - take newton step scaled by alpha
      x = x + std::get<0>(alpha_xy) * px;
      y = y + alpha_y * py;
      z = z + std::get<0>(alpha_xy) * pz;

      /// (dakre) 9 - compute residuals + mu again
      rd = dual_residual(y);
      rp = primal_residual(z, x);
      res.mu = mu(x, y, nu);

      rd_norm = linalg::L2<decltype(rd)>()(rd);
      rp_norm = linalg::L2<decltype(rp)>()(rp);

      dual_feas = rd_norm <= linalg::SolverTraits<T>::DUAL_FEASIBLE;
      primal_feas = rp_norm <= linalg::SolverTraits<T>::PRIMAL_FEASIBLE;
      gap_min = res.mu <= linalg::SolverTraits<T>::DUALITY_GAP;

      tag = get_tag(primal_feas, dual_feas, gap_min);
#ifdef AKREON_DEBUG
      trace[i].flags = (primal_feas ? (1 << 0) : 0) |
                       (dual_feas ? (1 << 1) : 0) | (gap_min ? (1 << 2) : 0);
#endif
    }
  }
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
  template <size_t idx = 0, size_t Start = 0>
  constexpr auto accumulate_params(Schur&& S, T&& nu, const FMat& F,
                                   const auto& K, const ConeVar& x,
                                   const ConeVar& y) const {
    if constexpr (idx >= NumCones)
      return std::tuple(S, nu);
    else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;

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
  template <size_t idx = 0, size_t Start = 0>
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
  template <size_t idx = 0, size_t Start = 0>
  constexpr auto alpha_max(T&& alpha_x, T&& alpha_y, const auto& K,
                           const ConeVar& x, const ConeVar& y,
                           const ConeVar& px, const ConeVar& py) const {
    if constexpr (idx >= NumCones) {
      return std::tuple(std::min(T(1), alpha_x), std::min(T(1), alpha_y));
    } else {
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
