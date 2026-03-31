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

#include "convex/convex_problem.hpp"
#include "convex/solver_traits.hpp"
#include "equality_coupling.hpp"
#include "newton_step.hpp"
#include "pd_info.hpp"
#include "pred_corr.hpp"
#include "residual_state.hpp"
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
 * @brief regularize
 *
 * case 1: no eq bump diagonal of M block by delta globally
 * case 2: eq in the loop but non-degenerate fall back to 1
 * case 3: eq in the loop but near-degenerate direction bump higher in this dir
 *
 * @param M (Mat)
 * @param A (EqMat)
 * @param reg (RegularizationConfig)
 *
 * @return M
 */
template <linalg::SquareMatrixLike Mat, linalg::MatrixLike EqMat>
constexpr auto regularize(
    Mat M, const EqMat& A,
    const config::RegularizationConfig<linalg::mat_type_t<EqMat>>& reg) {
  using T = linalg::mat_type_t<Mat>;
  constexpr size_t N = linalg::mat_cols_v<Mat>;
  constexpr size_t EqRows = linalg::mat_rows_v<EqMat>;
  constexpr size_t EqCols = linalg::mat_cols_v<EqMat>;

  const auto d = get_diag(M);
  T s{0};
  for (const auto& x : d) s = std::max(s, linalg::constexpr_abs(x));

  s = std::max(s, T(1));
  const auto delta = reg.global_eps * s;
  const auto eq_delta = reg.eq_eps * s;

  for (size_t j = 0; j < N; ++j) {
    T bump = delta;

    if constexpr (EqRows != 0 && EqCols != 0) {
      bool eq_active = false;
      if (j < EqCols) {
        for (size_t i = 0; i < EqRows; ++i) {
          if (linalg::constexpr_abs(A(i)(j)) > T(0)) {
            eq_active = true;
            break;
          }
        }
      }

      if (eq_active && linalg::constexpr_abs(d(j)) < eq_delta) bump = eq_delta;
    }

    M(j)(j) += bump;
  }

  return M;
}

/**
 * @brief Primal Dual Interior Point Method Solver
 */
template <ProblemLike Problem>
struct PrimalDualIPM {
  /// @brief alias and helpers to get types
  using Builder = prob_builder_t<Problem>;
  using DecVar = conic::builder_dom_t<Builder>;
  using ConeMap = decltype(std::declval<const Problem&>().constraint_map());
  using ConeMat = affine::affine_mat_t<ConeMap>;
  using EqMap = decltype(std::declval<const Problem&>().equality_map());
  using EqMat = affine::affine_mat_t<EqMap>;
  using T = linalg::space_type_t<DecVar>;

  /// @brief helpers for size of cones & F mat size & A mat rows
  static constexpr size_t NumCones = std::tuple_size_v<
      decltype(std::declval<const Problem&>().constraint_cones())>;
  static constexpr size_t M = linalg::mat_rows_v<ConeMat>;
  static constexpr size_t N = linalg::mat_cols_v<ConeMat>;

  static constexpr size_t EqRows = linalg::mat_rows_v<EqMat>;
  using NuVar = linalg::Vector<T, EqRows>;

  /// @brief useful types for PD-IPM
  using BlockMat = linalg::Matrix<T, N, N>;
  using Var = linalg::Vector<T, N>;
  using ConeVar = linalg::Vector<T, M>;
  using Step = NewtonStep<Var, ConeVar, NuVar>;
  using Residuals = ResidualState<Var, ConeVar, NuVar>;

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
      const Problem& prob,
      const PrimalDualIterate<Var, ConeVar, NuVar>& iterate) {
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
      const Problem& prob,
      const PrimalDualIterate<Var, ConeVar, NuVar>& iterate,
      const config::PrimalDualIPMConfig<T>& config_in)
      : config(config_in) {
    res.iterate = iterate;
    solve(prob, true);
  }

  /** @brief member vars */
  PrimalDualResult<Var, ConeVar, NuVar> res;  ///< PD-IPM result
  bool dual_feas{false};    ///< status flag for dual feasibility
  bool primal_feas{false};  ///< status flag for primal feasibility
  bool gap_min{false};      ///< status flag for duality gap minimized
#ifdef AKREON_DEBUG
  std::array<IterState<T, NumCones, M, N, EqRows>, PD_IPM_STEPS>
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
    const auto A = prob.equality_map().linear_part();
    const auto b = prob.equality_map().translation();
    const auto K = prob.constraint_cones();

    AK_ASSERT(prob.validate() && "submitted problem is not valid!");
    if (!prob.validate()) return;

    auto& z = res.iterate.z;
    auto& x = res.iterate.x;
    auto& y = res.iterate.y;
    auto& nu = res.iterate.nu;
    auto& rp = res.rp;
    auto& rd = res.rd;
    auto& req = res.req;
    auto& tag = res.tag;
    auto& i = res.num_iter;

    PredictorCorrector<NumCones, decltype(F), decltype(A), Var, ConeVar> pc;

    const auto barrier_param = seed_prob(K, x, y, warm_start);

    for (i = 0; i < config.steps && tag != config::SolverResult::OPTIMAL; ++i) {
#ifdef AKREON_DEBUG
      trace[i].reset_pc_trace();
#endif

      /// (dakre) 1 - accumulate and factor block
      const auto [M_blk, block_event] =
          accumulate_block(BlockMat{}, F, K, x, y);
      if (linalg::is_fatal_event(block_event)) {
        res.event = block_event;
        res.stalled = true;
        return;
      }
      const auto M_ldl = linalg::LDL(regularize(M_blk, A, config.reg));
      const auto [eq_opt, coupling_event] = make_equality_coupling(M_ldl, A);
      if (linalg::is_fatal_event(coupling_event) || !eq_opt.has_value()) {
        res.event = coupling_event;
        res.stalled = true;
        return;
      }
      const auto& eq = *eq_opt;

      /// (dakre) 2 - compute initial primal & dual residuals (negated)
      auto residuals = compute_residuals(F, A, g, c, b, z, nu, x, y);
      rd = residuals.rd;
      rp = residuals.rp;
      req = residuals.req;
      auto rd_norm = residuals.rd_norm;
      auto rp_norm = residuals.rp_norm;
      auto req_norm = residuals.req_norm;

      /// (dakre) 3 - predictor step
      const auto [pred_rhs, pred_step, pred_event] =
          pc.predictor(Var{}, K, M_ldl, eq, F, rp, rd, req, x, y
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
      const auto [corr_rhs, corr_step, corr_event] = pc.corrector(
          Var{}, pred_step.px, pred_step.py, K, M_ldl, eq, F, rp, rd, req, x, y
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

      /// (dakre) 5 - newton step (blended step \equiv convex combo)
      const auto step = blend_steps(pred_step, corr_step);

      /// (dakre) 6 - alpha max
      auto alpha_xy = alpha_max(T(1), T(1), K, x, y, step.px, step.py);

      /// (dakre) 7 - backtrack line search to acquire next iterate
      const auto accepted =
          backtrack_iterates(alpha_xy, K, F, A, g, c, b, step.pz, step.pnu,
                             step.px, step.py, z, nu, x, y);
      if (!accepted) {
        res.stalled = true;
        return;
      }

      res.mu = mu(x, y, barrier_param);
#ifdef AKREON_DEBUG
      trace[i].mu = res.mu;
      trace[i].rd_norm = rd_norm;
      trace[i].rp_norm = rp_norm;
      trace[i].req_norm = req_norm;
      trace[i].alpha_x = std::get<0>(alpha_xy);
      trace[i].alpha_y = std::get<1>(alpha_xy);
      trace[i].pz_norm = linalg::L2<decltype(step.pz)>()(step.pz);
      trace[i].pnu_norm = linalg::L2<decltype(step.pnu)>()(step.pnu);
      trace[i].px_norm = linalg::L2<decltype(step.px)>()(step.px);
      trace[i].py_norm = linalg::L2<decltype(step.py)>()(step.py);
#endif

      /// (dakre) 8 - compute residuals + mu again
      residuals = compute_residuals(F, A, g, c, b, z, nu, x, y);
      rd = residuals.rd;
      rp = residuals.rp;
      req = residuals.req;
      rd_norm = residuals.rd_norm;
      rp_norm = residuals.rp_norm;
      req_norm = residuals.req_norm;

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
   * @brief accumulate block
   *
   * [  0 F^T -A^T  0 ] [ p_z  ] = -[ r_d  ]
   * [  F 0    0   -I ] [ p_y  ]    [ r_p1 ]
   * [ -A 0    0    0 ] [ p_nu ]    [ r_p2 ]
   * [  0 I    0    H ] [ p_x  ]    [ r_c  ]
   *
   * 2x2 Saddle System
   * [ -F^T H F -A^T ] [ p_z; p_nu ] = - [ \hat{r}_d; r_p2 ]
   * [ -A        0   ]
   *
   * where M = sum(F_i^T H_i F_i)
   *
   * @param M Block matrix
   * @param F mapping from R^n -> conic space
   * @param K cartesian product of cones
   * @param x \in K
   * @param y \in K^*
   *
   * @return (S, nu)
   */
  template <size_t idx = 0, size_t Start = 0>
  constexpr auto accumulate_block(BlockMat&& M_blk, const ConeMat& F,
                                  const auto& K, const ConeVar& x,
                                  const ConeVar& y) const {
    if constexpr (idx >= NumCones)
      return std::tuple(std::move(M_blk), linalg::NumericalEvent::NONE);
    else {
      const auto Ki = std::get<idx>(K);
      using Slack = conic::cone_slack_t<decltype(Ki)>;

      constexpr size_t KiRows = linalg::vec_size_v<Slack>;
      constexpr size_t End = Start + KiRows;

      using ConePoint = linalg::Vector<T, End - Start>;
      ConePoint xk{}, yk{};

      for (size_t i = Start, j = 0; i < End; ++i, ++j)
        xk(j) = x(i), yk(j) = y(i);

      const auto [P, event] = Ki.H(xk, yk);
      if (linalg::is_fatal_event(event))
        return std::tuple(std::move(M_blk), event);

      const auto F_blk = linalg::get_block<Start, End, 0, N>(F);
      const auto Mi = linalg::transpose(F_blk) * P.apply(F_blk);
      M_blk = M_blk + Mi;

      return accumulate_block<idx + 1, End>(std::move(M_blk), F, K, x, y);
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
   * @param A (equality mat)
   * @param g (bias)
   * @param c (linear objective)
   * @param b (equality bias)
   * @param pz (newton step in z)
   * @param pnu (newton step in nu)
   * @param px (newton step in x)
   * @param pz (newton step in y)
   * @param z \in R^n
   * @param nu \in EQ dim
   * @param x \in K
   * @param y \in K^*
   *
   * @return T/F (indicates if backtracking found valid alpha)
   */
  constexpr bool backtrack_iterates(std::tuple<T, T>& alpha_xy, const auto& K,
                                    const ConeMat& F, const EqMat& A,
                                    const ConeVar& g, const Var& c,
                                    const NuVar& b, const Var& pz,
                                    const NuVar& pnu, const ConeVar& px,
                                    const ConeVar& py, Var& z, NuVar& nu,
                                    ConeVar& x, ConeVar& y) {
    T& ax = std::get<0>(alpha_xy);
    T& ay = std::get<1>(alpha_xy);
    ax = std::max(T(0), ax);
    ay = std::max(T(0), ay);

    const auto curr = compute_residuals(F, A, g, c, b, z, nu, x, y);

    constexpr auto zeta{1.05};

    for (size_t i = 0; i < config.backtrack_steps; ++i) {
      const auto z_tmp = z + ax * pz;
      const auto nu_tmp = nu + ax * pnu;
      const auto x_tmp = x + ax * px;
      const auto y_tmp = y + ay * py;

      const auto tmp =
          compute_residuals(F, A, g, c, b, z_tmp, nu_tmp, x_tmp, y_tmp);

      const bool inflation_check = tmp.rd_norm <= zeta * curr.rd_norm &&
                                   tmp.rp_norm <= zeta * curr.rp_norm &&
                                   tmp.req_norm <= zeta * curr.req_norm;

      if (check_functional_proximity(K, x_tmp, y_tmp) && inflation_check) {
        x = x_tmp;
        y = y_tmp;
        z = z_tmp;
        nu = nu_tmp;
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
   * F^T y - A^T \nu - c = rd
   *
   * @param F (global map)
   * @param A (equality map)
   * @param c (linear objective)
   * @param nu (equality multiplier)
   * @param y (dual var slack)
   *
   * @return rd
   */
  constexpr auto dual_residual(const ConeMat& F, const EqMat& A, const Var& c,
                               const NuVar& nu, const ConeVar& y) const {
    if constexpr (EqRows == 0)
      return linalg::transpose(F) * y - c;
    else
      return linalg::transpose(F) * y - linalg::transpose(A) * nu - c;
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
  constexpr auto primal_residual(const ConeMat& F, const ConeVar& g,
                                 const Var& z, const ConeVar& x) const {
    return F * z + g - x;
  }

  /**
   * @brief equality residual
   *
   * req = b - A z
   *
   * @param A (eq mat)
   * @param b (bias)
   * @param z (dec var)
   *
   * @return req
   */
  constexpr auto equality_residual(const EqMat& A, const NuVar& b,
                                   const Var& z) const {
    if constexpr (EqRows == 0)
      return NuVar{};
    else
      return b - A * z;
  }

  /**
   * @brief compute residual vectors and norms for the current iterate
   *
   * @param F (R^n -> conic space)
   * @param A (equality)
   * @param g (dual weights)
   * @param c (primal weights)
   * @param b (bias affine eq)
   * @param z (dec var)
   * @param nu (equality multiplier)
   * @param x \in K
   * @param y \in K^*
   */
  constexpr Residuals compute_residuals(const ConeMat& F, const EqMat& A,
                                        const ConeVar& g, const Var& c,
                                        const NuVar& b, const Var& z,
                                        const NuVar& nu, const ConeVar& x,
                                        const ConeVar& y) const {
    Residuals residuals{};
    residuals.rd = dual_residual(F, A, c, nu, y);
    residuals.rp = primal_residual(F, g, z, x);
    residuals.req = equality_residual(A, b, z);
    residuals.rd_norm = linalg::L2<Var>()(residuals.rd);
    residuals.rp_norm = linalg::L2<ConeVar>()(residuals.rp);
    residuals.req_norm = linalg::L2<NuVar>()(residuals.req);
    return residuals;
  }

  /**
   * @brief build equality-coupling object
   *
   * @param M_ldl (LDL^T(M))
   * @param A (equality map)
   *
   * @return equality coupling, event
   */
  constexpr auto make_equality_coupling(const auto& M_ldl,
                                        const EqMat& A) const {
    const auto& schur_config = config.schur;

    if constexpr (EqRows == 0) {
      using Coupling = NoEqualityCoupling<Var, NuVar>;
      return std::tuple(std::optional<Coupling>{Coupling{}},
                        linalg::NumericalEvent::NONE);
    } else if constexpr (EqRows == 1) {
      const auto [M_inv_AT, event] = M_ldl.apply(linalg::transpose(A));
      using Coupling = ScalarEqualityCoupling<EqMat, T, Var, NuVar>;
      if (linalg::is_fatal_event(event))
        return std::tuple(std::optional<Coupling>{}, event);

      const auto S = A * M_inv_AT;

      if (S(0)(0) < schur_config.tol)
        return std::tuple(std::optional<Coupling>{},
                          linalg::NumericalEvent::FACTORIZATION_FAILED);

      return std::tuple(std::optional<Coupling>{Coupling(A, S(0)(0))},
                        linalg::NumericalEvent::NONE);
    } else {
      const auto [M_inv_AT, event] = M_ldl.apply(linalg::transpose(A));
      using SchurFactor = decltype(linalg::LDL(A * M_inv_AT));
      using Coupling = SchurEqualityCoupling<EqMat, SchurFactor, Var, NuVar>;
      if (linalg::is_fatal_event(event))
        return std::tuple(std::optional<Coupling>{}, event);

      const auto S = A * M_inv_AT;
      const auto S_ldl = linalg::LDL(S, schur_config);
      if (!S_ldl.is_valid())
        return std::tuple(std::optional<Coupling>{},
                          linalg::NumericalEvent::FACTORIZATION_FAILED);

      return std::tuple(std::optional<Coupling>{Coupling(A, S_ldl)},
                        linalg::NumericalEvent::NONE);
    }
  }

  /**
   * @brief blend affine-scaling and centering newton directions
   *
   * @param pred_step (Newton dir for affine scaling dir)
   * @param corr_step (Newton dir for centering)
   *
   * @return blended (Newton dir)
   */
  constexpr Step blend_steps(const Step& pred_step,
                             const Step& corr_step) const {
    Step blended{};
    blended.px = rho * pred_step.px + (T(1) - rho) * corr_step.px;
    blended.py = rho * pred_step.py + (T(1) - rho) * corr_step.py;
    blended.pz = rho * pred_step.pz + (T(1) - rho) * corr_step.pz;
    blended.pnu = rho * pred_step.pnu + (T(1) - rho) * corr_step.pnu;
    return blended;
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
