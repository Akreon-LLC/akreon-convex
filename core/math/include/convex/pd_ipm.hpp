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
#include "pred_corr.hpp"
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
constexpr auto mu(const ConeVar& x, const ConeVar& y,
                  const linalg::vec_type_t<ConeVar>& nu) {
  return linalg::dot(x, y) / nu;
}

/**
 * @brief PrimalDualIterate
 */
template <linalg::VectorLike Var, linalg::VectorLike ConeVar>
struct PrimalDualIterate {
  Var z{};      ///< decision var \in Rn + aux lifts
  ConeVar x{};  ///< primal cone var \in K
  ConeVar y{};  ///< dual cone var \in K^*
};

/**
 * @brief PrimalDualResult
 */
template <linalg::VectorLike Var, linalg::VectorLike ConeVar>
struct PrimalDualResult {
  using T = linalg::vec_type_t<Var>;

  config::SolverResult tag{};  ///< tag for result of pd-ipm
  linalg::NumericalEvent event{linalg::NumericalEvent::NONE};
  bool stalled{false};  ///< backtracking status (stalled or not)
  T mu{};               ///< normalized duality gap
  T convex_obj{};       ///< convex objective value c^T z
  T dual_obj{};         ///< dual objective value -g^T y
  Var rd{};             ///< dual residuals
  ConeVar rp{};         ///< primal residuals
  PrimalDualIterate<Var, ConeVar> iterate;  ///< primal dual iterate
  size_t num_iter{};                        ///< total number of iter
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
   * @param config (PrimalDualIPMConfig)
   */
  constexpr explicit PrimalDualIPM(
      const Problem& prob, const config::PrimalDualIPMConfig<T>& config_in)
      : config(config_in) {
    solve(prob);
  }

  /**
   * @brief CTOR (warm start)
   *
   * @param prob (ConvexProblem)
   * @param iterate (PrimalDualIterate)
   */
  constexpr explicit PrimalDualIPM(
      const Problem& prob, const PrimalDualIterate<Var, ConeVar>& iterate) {
    res.iterate = iterate;
    solve(prob, true);
  }

  /**
   * @brief CTOR (warm start)
   *
   * @param prob (ConvexProblem)
   * @param iterate (PrimalDualIterate)
   * @param config (PrimalDualIPMConfig)
   */
  constexpr explicit PrimalDualIPM(
      const Problem& prob, const PrimalDualIterate<Var, ConeVar>& iterate,
      const config::PrimalDualIPMConfig<T>& config_in)
      : config(config_in) {
    res.iterate = iterate;
    solve(prob, true);
  }

  /** @brief member vars */
  PrimalDualResult<Var, ConeVar> res;  ///< PD-IPM result
  bool dual_feas{false};               ///< status flag for dual feasibility
  bool primal_feas{false};             ///< status flag for primal feasibility
  bool gap_min{false};  ///< status flag for duality gap minimized
#ifdef AKREON_DEBUG
  std::array<IterState<T, NumCones, M, N>, PD_IPM_STEPS>
      trace;  ///< debug iteration trace
#endif

 private:
  /**
   * @brief solve
   *
   * @param prob (ConvexProblem)
   * @param warm_start (bool)
   *
   * @return result
   */
  constexpr void solve(const Problem& prob, bool warm_start = false) {
    const auto c = linalg::as_vector(prob.linear_objective());
    const auto g = prob.constraint_map().translation();
    const auto F = prob.constraint_map().linear_part();
    const auto K = prob.constraint_cones();

    AK_ASSERT(prob.validate() && "submitted problem is not valid!");
    if (!prob.validate()) return;

    auto& z = res.iterate.z;
    auto& x = res.iterate.x;
    auto& y = res.iterate.y;
    auto& rp = res.rp;
    auto& rd = res.rd;
    auto& tag = res.tag;
    auto& i = res.num_iter;

    PredictorCorrector<NumCones, decltype(F), Var, ConeVar> pc;

    const auto nu = seed_prob(K, x, y, warm_start);

    for (i = 0; i < config.steps && tag != config::SolverResult::OPTIMAL; ++i) {
      ConeVar px_pred{}, px_corr{}, py_pred{}, py_corr{};
      Var pz_pred{}, pz_corr{};
#ifdef AKREON_DEBUG
      trace[i].reset_pc_trace();
#endif

      /// (dakre) 1 - accumulate and factor schur
      const auto [S, schur_event] = accumulate_schur(Schur{}, F, K, x, y);
      if (linalg::is_fatal_event(schur_event)) {
        res.event = schur_event;
        res.stalled = true;
        return;
      }
      const auto ldl = linalg::LDL(linalg::regularize(S));

      /// (dakre) 2 - compute initial primal & dual residuals (negated)
      rd = dual_residual(F, c, y);
      rp = primal_residual(F, g, z, x);
      auto rd_norm = linalg::L2<decltype(rd)>()(rd);
      auto rp_norm = linalg::L2<decltype(rp)>()(rp);

      /// (dakre) 3 - predictor step
      const auto [pred_rhs, pred_event] =
          pc.predictor(Var{}, K, ldl, F, rp, rd, x, y, pz_pred, px_pred, py_pred
#ifdef AKREON_DEBUG
                       ,
                       trace[i]
#endif
          );
      static_cast<void>(pred_rhs);
      if (linalg::is_fatal_event(pred_event)) {
        res.event = pred_event;
        res.stalled = true;
        return;
      }

      /// (dakre) 4 - corrector step
      const auto [corr_rhs, corr_event] =
          pc.corrector(Var{}, px_pred, py_pred, K, ldl, F, rp, rd, x, y,
                       pz_corr, px_corr, py_corr
#ifdef AKREON_DEBUG
                       ,
                       trace[i]
#endif
          );
      static_cast<void>(corr_rhs);
      if (linalg::is_fatal_event(corr_event)) {
        res.event = corr_event;
        res.stalled = true;
        return;
      }

      /// (dakre) 5 - newton step (blended step)
      const auto px = rho * px_pred + (1.0 - rho) * px_corr;
      const auto py = rho * py_pred + (1.0 - rho) * py_corr;
      const auto pz = rho * pz_pred + (1.0 - rho) * pz_corr;

      /// (dakre) 6 - alpha max
      auto alpha_xy = alpha_max(T(1), T(1), K, x, y, px, py);

      /// (dakre) 7 - backtrack line search to acquire next iterate
      backtrack_iterates(alpha_xy, K, F, g, c, pz, px, py, z, x, y);

      res.mu = mu(x, y, nu);
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
      return std::tuple(std::move(S), linalg::NumericalEvent::NONE);
    else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;

      constexpr size_t KiRows = linalg::vec_size_v<Slack>;
      constexpr size_t End = Start + KiRows;

      using ConePoint = linalg::Vector<T, End - Start>;
      ConePoint xk{}, yk{};

      for (size_t i = Start, j = 0; i < End; ++i, ++j)
        xk(j) = x(i), yk(j) = y(i);

      const auto [Hes, event] = Ki.H(xk, yk);
      if (linalg::is_fatal_event(event)) return std::tuple(std::move(S), event);

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
   * @brief seed problem
   *
   * Initialize x \in K and y \in K^* with initial primal & dual points in
   * each cone
   *
   * @param K cartesian product of cones
   * @param x \in K
   * @param y \in K^*
   * @param warm_start (bool)
   *
   * @return nu
   */
  template <size_t idx = 0, size_t Start = 0>
  constexpr T seed_prob(const auto& K, ConeVar& x, ConeVar& y,
                        const bool& warm_start) const {
    if constexpr (idx >= NumCones)
      return T(0);
    else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;

      constexpr size_t KiRows = linalg::vec_size_v<Slack>;
      constexpr size_t End = Start + KiRows;

      if (warm_start) {
        using ConePoint = linalg::Vector<T, End - Start>;
        ConePoint xk{}, yk{};
        for (size_t i = Start, j = 0; i < End; ++i, ++j)
          xk(j) = x(i), yk(j) = y(i);

        if (Ki.in_cone(xk) && Ki.in_dual_cone(yk))
          return Ki.nu() + seed_prob<idx + 1, End>(K, x, y, warm_start);
      }

      const auto x0 = Ki.x0(), y0 = Ki.y0();
      for (size_t i = Start, j = 0; i < End; ++i, ++j)
        x(i) = x0(j), y(i) = y0(j);

      return Ki.nu() + seed_prob<idx + 1, End>(K, x, y, warm_start);
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
      const auto [gamma_F, event] = Ki.gamma_F(xk, yk);
      bool rv = Ki.in_cone(xk) && Ki.in_dual_cone(yk) &&
                !linalg::is_fatal_event(event) && gamma_F < beta_limit;
      return rv && check_functional_proximity<idx + 1, End>(K, x, y);
    }
  }

  /**
   * @brief backtrack alpha to acquire next iterate
   *
   * @param alpha_xy (tuple alpha step in x & y dir)
   * @param K (cartesian product of cones)
   * @param F (cone map)
   * @param g (bias)
   * @param c (linear objective)
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
                                    const FMat& F, const ConeVar& g,
                                    const Var& c, const Var& pz,
                                    const ConeVar& px, const ConeVar& py,
                                    Var& z, ConeVar& x, ConeVar& y) {
    T& ax = std::get<0>(alpha_xy);
    T& ay = std::get<1>(alpha_xy);
    ax = std::max(T(0), ax);
    ay = std::max(T(0), ay);

    const auto rd_curr = linalg::L2<Var>()(dual_residual(F, c, y));
    const auto rp_curr = linalg::L2<ConeVar>()(primal_residual(F, g, z, x));

    constexpr auto zeta{1.05};

    for (size_t i = 0; i < config.backtrack_steps; ++i) {
      const auto z_tmp = z + ax * pz;
      const auto x_tmp = x + ax * px;
      const auto y_tmp = y + ay * py;

      const auto rd_tmp = linalg::L2<Var>()(dual_residual(F, c, y_tmp));
      const auto rp_tmp =
          linalg::L2<ConeVar>()(primal_residual(F, g, z_tmp, x_tmp));

      if (check_functional_proximity(K, x_tmp, y_tmp) &&
          rd_tmp <= zeta * rd_curr && rp_tmp <= zeta * rp_curr) {
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
