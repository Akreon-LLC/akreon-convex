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
#ifdef AKREON_DEBUG
#include "pd_ipm_debug.hpp"
#endif

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
constexpr auto mu_global(const ConeVar& x, const ConeVar& y,
                         const linalg::vec_type_t<ConeVar>& nu) {
  return linalg::dot(x, y) / nu;
}

/**
 * @brief PrimalDualResult
 */
template <linalg::VectorLike Var, linalg::VectorLike ConeVar>
struct PrimalDualResult {
  using T = linalg::vec_type_t<Var>;

  config::SolverResult tag{};  ///< tag for result of pd-ipm
  bool stalled{false};         ///< backtracking status (stalled or not)
  T mu{};                      ///< normalized duality gap
  T convex_obj{};              ///< convex objective value c^T z
  T dual_obj{};                ///< dual objective value -g^T y
  Var rd{};                    ///< dual residuals
  ConeVar rp{};                ///< primal residuals
  Var z{};                     ///< decision var \in Rn + aux lifts
  ConeVar x{};                 ///< primal cone var \in K
  ConeVar y{};                 ///< dual cone var \in K^*
  size_t num_iter{};           ///< total number of iter
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
  constexpr Var predictor(Var&& pz_rhs, const auto& K, const auto& S_inv,
                          const Mat& F, const ConeVar& rp, const Var& rd,
                          const ConeVar& x, const ConeVar& y, Var& pz,
                          ConeVar& px, ConeVar& py
#ifdef AKREON_DEBUG
                          ,
                          IterState<T, NumCones>* trace_state = nullptr
#endif
  ) const {
    if constexpr (idx >= NumCones)
      return pz_rhs;
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

      const auto Hes = Ki.H(xk, yk);
      const auto F_blk = linalg::get_block<Start, End, 0, N>(F);
      const auto F_blk_T = linalg::transpose(F_blk);

      ConePoint rhs_c = yk;

      pz_rhs = pz_rhs - F_blk_T * Hes.apply(rpk) - F_blk_T * rhs_c;
#ifdef AKREON_DEBUG
      if (trace_state != nullptr) {
        auto& cone_trace = trace_state->predictor[idx];
        cone_trace.mu = Ki.trace(xk, yk) / Ki.nu();
        cone_trace.rhs_c = linalg::L2<ConePoint>()(rhs_c);
        cone_trace.pz_rhs = linalg::L2<Var>()(pz_rhs);
        cone_trace.cone_type = detail::cone_type_name_v<decltype(Ki)>;
        cone_trace.active = true;
      }
#endif

      const auto new_pz_rhs = predictor<idx + 1, End>(
          std::move(pz_rhs), K, S_inv, F, rp, rd, x, y, pz, px, py
#ifdef AKREON_DEBUG
          ,
          trace_state
#endif
      );

      if constexpr (idx + 1 == NumCones) pz = S_inv.solve(new_pz_rhs + rd);

      const auto px_local = F_blk * pz + rpk;
      const auto py_local = T(-1) * rhs_c - Hes.apply(px_local);

      for (size_t i = Start, j = 0; i < End; ++i, ++j) {
        px(i) = px_local(j), py(i) = py_local(j);
      }

      return new_pz_rhs;
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
  constexpr Var corrector(Var&& pz_rhs, const ConeVar& px_aff,
                          const ConeVar& py_aff, const auto& K,
                          const auto& S_inv, const Mat& F, const ConeVar& rp,
                          const Var& rd, const ConeVar& x, const ConeVar& y,
                          Var& pz, ConeVar& px, ConeVar& py
#ifdef AKREON_DEBUG
                          ,
                          IterState<T, NumCones>* trace_state = nullptr
#endif
  ) const {
    if constexpr (idx >= NumCones)
      return pz_rhs;
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

      const auto Hes = Ki.H(xk, yk);
      const auto F_blk = linalg::get_block<Start, End, 0, N>(F);
      const auto F_blk_T = linalg::transpose(F_blk);

      T mu_k = Ki.trace(xk, yk) / Ki.nu();
      ConePoint rhs_c = yk + mu_k * Ki.grad(xk) + Ki.circ(px_aff_k, py_aff_k);

      pz_rhs = pz_rhs - F_blk_T * Hes.apply(rpk) - F_blk_T * rhs_c;
#ifdef AKREON_DEBUG
      if (trace_state != nullptr) {
        auto& cone_trace = trace_state->corrector[idx];
        cone_trace.mu = mu_k;
        cone_trace.rhs_c = linalg::L2<ConePoint>()(rhs_c);
        cone_trace.pz_rhs = linalg::L2<Var>()(pz_rhs);
        cone_trace.cone_type = detail::cone_type_name_v<decltype(Ki)>;
        cone_trace.active = true;
      }
#endif

      const auto new_pz_rhs =
          corrector<idx + 1, End>(std::move(pz_rhs), px_aff, py_aff, K, S_inv,
                                  F, rp, rd, x, y, pz, px, py
#ifdef AKREON_DEBUG
                                  ,
                                  trace_state
#endif
          );

      if constexpr (idx + 1 == NumCones) pz = S_inv.solve(new_pz_rhs + rd);

      const auto px_local = F_blk * pz + rpk;
      const auto py_local = T(-1) * rhs_c - Hes.apply(px_local);

      for (size_t i = Start, j = 0; i < End; ++i, ++j) {
        px(i) = px_local(j), py(i) = py_local(j);
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

  /// @brief damping param using for alpha steps in predictor corrector
  static constexpr T rho{linalg::ln2_v<T>};

  /// @brief beta F(\beta) neighborhood of central path (within feasibility)
  static constexpr T beta{1 - linalg::ln2_v<T>};

  /**
   * @brief CTOR
   *
   * @param prob (ConvexProblem)
   */
  constexpr explicit PrimalDualIPM(const Problem& prob) { solve(prob); }

  /**
   * @brief CTOR
   *
   * @param prob (ConvexProblem)
   * @param config (PrimalDualIPM Config)
   */
  constexpr explicit PrimalDualIPM(
      const Problem& prob, const config::PrimalDualIPMConfig<T>& config_in)
      : config(config_in) {
    solve(prob);
  }

  /** @brief member vars */
  PrimalDualResult<Var, ConeVar> res;  ///< PD-IPM result
  bool dual_feas{false};               ///< status flag for dual feasibility
  bool primal_feas{false};             ///< status flag for primal feasibility
  bool gap_min{false};  ///< status flag for duality gap minimized
#ifdef AKREON_DEBUG
  std::array<IterState<T, NumCones>, PD_IPM_STEPS>
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

    AK_ASSERT(prob.validate() && "submitted problem is not valid!");

    auto& z = res.z;
    auto& x = res.x;
    auto& y = res.y;
    auto& rp = res.rp;
    auto& rd = res.rd;
    auto& tag = res.tag;
    auto& i = res.num_iter;

    PredictorCorrector<NumCones, decltype(F), Var, ConeVar> pc;

    seed_cone_vars(K, x, y);

    for (i = 0; i < config.steps && tag != config::SolverResult::OPTIMAL; ++i) {
      ConeVar px_pred{}, px_corr{}, py_pred{}, py_corr{};
      Var pz_pred{}, pz_corr{};
#ifdef AKREON_DEBUG
      trace[i].reset_pc_trace();
      auto* iter_trace = &trace[i];
#endif

      /// (dakre) 1 - accumulate and factor schur
      const auto S = accumulate_schur(Schur{}, F, K, x, y);
      const auto ldl = linalg::LDL(S);

      /// (dakre) 2 - compute initial primal & dual residuals (negated)
      rd = dual_residual(F, c, y);
      rp = primal_residual(F, g, z, x);
      auto rd_norm = linalg::L2<decltype(rd)>()(rd);
      auto rp_norm = linalg::L2<decltype(rp)>()(rp);

      /// (dakre) 3 - predictor step
      pc.predictor(Var{}, K, ldl, F, rp, rd, x, y, pz_pred, px_pred, py_pred
#ifdef AKREON_DEBUG
                   ,
                   iter_trace
#endif
      );

      /// (dakre) 4 - corrector step
      pc.corrector(Var{}, px_pred, py_pred, K, ldl, F, rp, rd, x, y, pz_corr,
                   px_corr, py_corr
#ifdef AKREON_DEBUG
                   ,
                   iter_trace
#endif
      );

      /// (dakre) 5 - newton step
      const auto px = px_pred + rho * px_corr;
      const auto py = py_pred + rho * py_corr;
      const auto pz = pz_pred + rho * pz_corr;

      /// (dakre) 6 - alpha max
      auto alpha_xy = alpha_max(T(1), T(1), K, x, y, px, py);

      /// (dakre) 7 - backtrack line search to acquire next iterate
      backtrack_iterates(alpha_xy, K, pz, px, py, z, x, y);

      const auto mu_next = mu(K, x, y);
      res.mu = std::get<0>(mu_next) / std::get<1>(mu_next);
#ifdef AKREON_DEBUG
      trace[i].mu = res.mu;
      trace[i].rd_norm = rd_norm;
      trace[i].rp_norm = rp_norm;
      trace[i].alpha_x = std::get<0>(alpha_xy);
      trace[i].alpha_y = std::get<1>(alpha_xy);
      trace[i].pz_norm = linalg::L2<decltype(pz)>()(pz);
      trace[i].px_norm = linalg::L2<decltype(px)>()(px);
      trace[i].py_norm = linalg::L2<decltype(py)>()(py);
#endif

      /// (dakre) 8 - compute residuals + mu again
      rd = dual_residual(F, c, y);
      rp = primal_residual(F, g, z, x);

      rd_norm = linalg::L2<decltype(rd)>()(rd);
      rp_norm = linalg::L2<decltype(rp)>()(rp);

      dual_feas = rd_norm <= config.dual_feas_tol;
      primal_feas = rp_norm <= config.primal_feas_tol;
      gap_min = res.mu >= T(0) && res.mu <= config.duality_gap_tol;

      tag = get_tag();
#ifdef AKREON_DEBUG
      trace[i].tag = tag;
#endif
    }

    res.convex_obj = linalg::dot(c, z);
    res.dual_obj = -linalg::dot(g, y);
  }

  /**
   * @brief accumulate params
   *
   * @param K (cartesian product of cones)
   * @param x \in K
   * @param y \in K^*
   *
   * @return (duality gap, nu)
   */
  template <size_t idx = 0, size_t Start = 0>
  constexpr auto mu(const auto& K, const ConeVar& x, const ConeVar& y) const {
    if constexpr (idx >= NumCones)
      return std::tuple(T(0), T(0));
    else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;

      constexpr size_t KiRows = linalg::vec_size_v<Slack>;
      constexpr size_t End = Start + KiRows;

      using ConePoint = linalg::Vector<T, End - Start>;
      ConePoint xk{}, yk{};

      for (size_t i = Start, j = 0; i < End; ++i, ++j)
        xk(j) = x(i), yk(j) = y(i);

      auto params = mu<idx + 1, End>(K, x, y);

      auto gap = std::get<0>(params);
      auto nu = std::get<1>(params);

      gap += Ki.trace(xk, yk);
      nu += Ki.nu();

      return std::tuple(gap, nu);
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
   * @param F mapping from R^n -> conic space
   * @param K cartesian product of cones
   * @param x \in K
   * @param y \in K^*
   *
   * @return (S, nu)
   */
  template <size_t idx = 0, size_t Start = 0>
  constexpr auto accumulate_schur(Schur&& S, const FMat& F, const auto& K,
                                  const ConeVar& x, const ConeVar& y) const {
    if constexpr (idx >= NumCones)
      return S;
    else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;

      constexpr size_t KiRows = linalg::vec_size_v<Slack>;
      constexpr size_t End = Start + KiRows;

      using ConePoint = linalg::Vector<T, End - Start>;
      ConePoint xk{}, yk{};

      for (size_t i = Start, j = 0; i < End; ++i, ++j)
        xk(j) = x(i), yk(j) = y(i);

      const auto Hes = Ki.H(xk, yk);

      const auto F_blk = get_block<Start, End, 0, N>(F);
      const linalg::SchurKKT<linalg::operator_mat_t<decltype(Hes)>,
                             decltype(F_blk)>
          kkt;
      const auto Si = kkt.make(Hes, F_blk);
      S = S + Si;

      return accumulate_schur<idx + 1, End>(std::move(S), F, K, x, y);
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

      const auto x0 = Ki.x0(), y0 = Ki.y0();
      for (size_t i = Start, j = 0; i < End; ++i, ++j)
        x(i) = x0(j), y(i) = y0(j);

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
      const auto ax = std::max(T(0), std::min(T(1), alpha_x));
      const auto ay = std::max(T(0), std::min(T(1), alpha_y));
      return std::tuple(ax, ay);
    } else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;
      constexpr size_t KiRows = linalg::vec_size_v<Slack>;
      constexpr size_t End = Start + KiRows;

      using ConePoint = linalg::Vector<T, End - Start>;
      ConePoint xk{}, yk{}, pxk{}, pyk{};

      for (size_t i = Start, j = 0; i < End; ++i, ++j)
        xk(j) = x(i), yk(j) = y(i), pxk(j) = px(i), pyk(j) = py(i);

      alpha_x = std::min(Ki.alpha_max(xk, pxk), alpha_x);
      alpha_y = std::min(Ki.alpha_max_dual(yk, pyk), alpha_y);

      return alpha_max<idx + 1, End>(std::move(alpha_x), std::move(alpha_y), K,
                                     x, y, px, py);
    }
  }

  /**
   * @brief dual residual
   *
   * F^T y - c = rd
   *
   * @param F (global map)
   * @param c (linear objective)
   * @param y (dual var slack)
   *
   * @return rd
   */
  static constexpr auto dual_residual(const FMat& F, const Var& c,
                                      const ConeVar& y) {
    return linalg::transpose(F) * y - c;
  }

  /**
   * @brief primal residual
   *
   * rp = F z + g - x
   *
   * @param F (global map)
   * @param g (dual objective / bias)
   * @param z (dec var)
   * @param x (primal var slack)
   *
   * @return rp
   */
  static constexpr auto primal_residual(const FMat& F, const ConeVar& g,
                                        const Var& z, const ConeVar& x) {
    return F * z + g - x;
  }

  /**
   * @brief in cones membership check
   *
   * check x \in K and y \in K^*
   *
   * @param K (cartesian product of cones)
   * @param x \in K
   * @param y \in K^*
   *
   * @return T/F
   */
  template <size_t idx = 0, size_t Start = 0>
  constexpr bool check_functional_proximity(const auto& K, const ConeVar& x,
                                            const ConeVar& y) const {
    if constexpr (idx >= NumCones) {
      return true;
    } else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;
      constexpr size_t KiRows = linalg::vec_size_v<Slack>;
      constexpr size_t End = Start + KiRows;

      using ConePoint = linalg::Vector<T, End - Start>;
      ConePoint xk{}, yk{};
      for (size_t i = Start, j = 0; i < End; ++i, ++j)
        xk(j) = x(i), yk(j) = y(i);

      const auto beta_limit = beta + config.Delta;
      bool rv = Ki.in_cone(xk) && Ki.in_dual_cone(yk) &&
                Ki.gamma_F(xk, yk) < beta_limit;
      return rv && check_functional_proximity<idx + 1, End>(K, x, y);
    }
  }

  /**
   * @brief backtrack alpha to acquire next iterate
   *
   * @param alpha_xy (tuple alpha step in x & y dir)
   * @param K (cartesian product of cones)
   * @param pz (newton step in z)
   * @param px (newton step in x)
   * @param pz (newton step in y)
   * @param z \in R^n
   * @param x \in K
   * @param y \in K^*
   *
   * @return T/F (indicates if backtracking found valid alpha)
   */
  constexpr bool backtrack_iterates(std::tuple<T, T>& alpha_xy, const auto& K,
                                    const Var& pz, const ConeVar& px,
                                    const ConeVar& py, Var& z, ConeVar& x,
                                    ConeVar& y) {
    T& ax = std::get<0>(alpha_xy);
    T& ay = std::get<1>(alpha_xy);
    ax = std::max(T(0), ax);
    ay = std::max(T(0), ay);
    for (size_t i = 0; i < config.backtrack_steps; ++i) {
      const auto z_tmp = z + ax * pz;
      const auto x_tmp = x + ax * px;
      const auto y_tmp = y + ay * py;

      if (check_functional_proximity(K, x_tmp, y_tmp)) {
        x = x_tmp;
        y = y_tmp;
        z = z_tmp;
        return true;
      }

      ax *= rho;
      ay *= rho;
    }
    return false;
  }

  /**
   * @brief get tag for current iterate
   *
   * @return tag (SolverResult)
   */
  constexpr auto get_tag() const {
    config::SolverResult tag;
    if (primal_feas && dual_feas && gap_min)
      tag = config::SolverResult::OPTIMAL;
    else if (primal_feas && dual_feas)
      tag = config::SolverResult::PRIMAL_DUAL_FEASIBLE;
    else if (primal_feas && gap_min)
      tag = config::SolverResult::PRIMAL_OPTIMAL;
    else if (dual_feas && gap_min)
      tag = config::SolverResult::DUAL_OPTIMAL;
    else if (primal_feas)
      tag = config::SolverResult::PRIMAL_FEASIBLE;
    else if (dual_feas)
      tag = config::SolverResult::DUAL_FEASIBLE;
    else if (gap_min)
      tag = config::SolverResult::INFEASIBLE_OPTIMAL;
    else
      tag = config::SolverResult::INFEASIBLE_OR_UNBOUNDED;
    return tag;
  }

  /** @brief member vars */
  config::PrimalDualIPMConfig<T> config;  ///< PD-IPM config
};

}  // namespace convex
