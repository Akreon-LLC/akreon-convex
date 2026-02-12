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
#include "linalg/operator/augmented_kkt.hpp"

namespace convex {

/**
 * @brief PrimalDualResult
 */
template <linalg::VectorLike DecVar, linalg::VectorLike ConeVar>
struct PrimalDualResult {
  /**
   * @brief CTOR
   *
   * @param result SolverTraits result
   */
  constexpr explicit PrimalDualResult(const linalg::SolverResult& res_in)
      : result(res_in) {}

  /**
   * @brief CTOR
   *
   * @param z dec var
   * @param x primal var
   * @param s dual var
   */
  constexpr explicit PrimalDualResult(const linalg::SolverResult& res_in,
                                      const DecVar& z_in, const ConeVar& x_in,
                                      const ConeVar& s_in)
      : result(res_in), z(z_in), x(x_in), s(s_in) {}

  /** @brief member vars */
  linalg::SolverResult result;  ///< result (solver)
  DecVar z;                     ///< decision var \in Rn / E
  ConeVar x;                    ///< primal cone var \in K
  ConeVar s;                    ///< dual cone var \in K^*
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
  static constexpr size_t N = linalg::space_dim_v<DecVar>;
  static constexpr size_t M = linalg::mat_rows_v<
      affine::affine_mat_t<conic::cone_block_map_t<builder_cb_t<Builder>>>>;
  static constexpr size_t NumCones =
      conic::cone_block_num_cones_v<builder_cb_t<Builder>>;
  using Schur = linalg::Matrix<T, N, N>;
  using Var = linalg::Vector<T, N>;
  using ConeVar = linalg::Vector<T, M>;

  /// @brief static constexpr for neighborhood size Beta < 1 - ln2
  static constexpr T Beta{0.1};

  /**
   * @brief solve
   *
   * @param prob ConvexProblem
   *
   * @return result
   */
  constexpr auto solve(
      const ConvexProblem<Builder, StageType::PROBLEM_SET>& prob) const {
    if (!prob.validate())
      return PrimalDualResult<Var, ConeVar>(linalg::SolverResult::INVALID);

    linalg::Vector<T, M> x{}, s{}, px_pred{}, px_corr{}, py_pred{}, py_corr{};
    Var z{}, pz_rhs_pred{}, pz_rhs_corr{}, pz_pred{}, pz_corr{};

    Schur S{};

    const auto c = prob.linear_objective();
    const auto g = prob.constraint_map().translation();
    const auto F = prob.constraint_map().linear_part();
    const auto K = prob.constraint_cones();

    /**
     * @brief process cones for predictor corrector method
     *
     * \tilde{KKT} = [ 0  0   -F^T ] [ pz; px; qy ] = [ rd; rc; rp ]
     *               [ 0  H_x I    ]
     *               [ -F I   0    ]
     *
     * dual / stationarity residual
     * rd = F^T y + c
     *
     * primal residual
     * rp = F z - x - g
     *
     * complementarity residual
     * rc = s or s + mu(x,s) F'(x)
     *
     * \tilde{S} = F^T H_x F
     * qy = -py
     *
     * preorder
     * (1) compute hessian at current iterate
     * (2) preorder accumuate schur (S) + rhs for pz calculation
     * postorder
     * (3) only once factor schur (S) and compute pz (global \in Rn)
     * (4) for each cone solve qy = -py & px
     */
    auto pred_corr_method = [&]<size_t idx = 0, size_t Start = 0>(
                                this auto&& self, const auto& rd,
                                const auto& rp) {
      if constexpr (idx >= NumCones)
        return false;
      else {
        const auto Ki = std::get<idx>(K);
        using Slack = conic::cone_slack_t<decltype(Ki)>;

        constexpr size_t KiRows = linalg::vec_size_v<Slack>;
        constexpr size_t End = Start + KiRows;

        if (conic::is_zoc_v<decltype(Ki)>)
          return self.template operator()<idx + 1, Start + KiRows>(rd, rp);

        if constexpr (Start == 0) {
          const auto x0 = Ki.x0();
          for (size_t i = Start, j = 0; i < End; ++i, ++j) x(i) = x0(j);
          const auto s0 = Ki.s0();
          for (size_t i = Start, j = 0; i < End; ++i, ++j) s(i) = x0(j);
        }

        using ConePoint = linalg::Vector<T, End - Start>;
        ConePoint xk{}, sk{};

        for (size_t i = Start, j = 0; i < End; ++i, ++j)
          xk(j) = x(i), sk(j) = s(i);

        const auto Hes = Ki.H(xk, sk);

        const auto F_blk = get_block<Start, Start + KiRows, 0, N>(F);
        const linalg::AugmentedKKT<decltype(Hes.H), decltype(F_blk)> kkt;
        const auto Si = kkt.make_schur(Hes, F_blk);
        S += Si;

        const auto FT = linalg::transpose(F_blk);
        const auto rc_pred = sk;
        const auto rc_corr = sk + Ki.mu(xk, sk) * Ki.grad(xk);

        pz_rhs_pred += (FT * rc_pred - FT * (Hes.apply(rp) + rd));
        pz_rhs_corr += (FT * rc_corr - FT * (Hes.apply(rp) + rd));

        const auto factored =
            self.template operator()<idx + 1, Start + KiRows>(rd, rp);

        if (!factored) {
          const auto ldl = linalg::LDL(S);
          pz_pred = ldl.solve(pz_rhs_pred);
          pz_corr = ldl.solve(pz_rhs_corr);
        }

        const auto py_pred_tmp = Hes.apply(F_blk * pz_pred + rd) - rc_pred;
        const auto py_corr_tmp = Hes.apply(F_blk * pz_corr + rd) - rc_corr;
        const auto px_pred_tmp = F_blk * pz_pred + rd;
        const auto px_corr_tmp = F_blk * pz_corr + rd;

        for (size_t i = Start, j = 0; i < End; ++i, ++j) {
          px_pred(i) = px_pred_tmp(j);
          px_corr(i) = px_corr_tmp(j);
          py_pred(i) = py_pred_tmp(j);
          py_corr(i) = py_corr_tmp(j);
        }

        return true;
      }
    };

    auto backtrack = [&]<size_t idx = 0, size_t Start = 0>(
                         this auto&& self, T alpha, const Var& px,
                         const Var& py) {
      if constexpr (idx >= NumCones)
        return alpha;
      else {
        const auto Ki = std::get<idx>(K);
        using Slack = conic::cone_slack_t<decltype(Ki)>;
        constexpr size_t KiRows = linalg::vec_size_v<Slack>;
        constexpr size_t End = Start + KiRows;

        if (conic::is_zoc_v<decltype(Ki)>)
          return self.template operator()<idx + 1, Start + KiRows>(alpha, px,
                                                                   py);

        using ConePoint = linalg::Vector<T, End - Start>;
        ConePoint xk{}, sk{}, pxk{}, pyk{};

        for (size_t i = Start, j = 0; i < End; ++i, ++j)
          xk(j) = x(i), sk(j) = s(i), pxk(j) = px(i), pyk(j) = py(i);

        const T alpha_x = Ki.alpha_max(xk, pxk);
        const T alpha_s = Ki.alpha_max(sk, pyk);

        alpha = std::min(alpha, std::min(alpha_x, alpha_s));

        auto xkb = xk - alpha * pxk;
        auto skb = sk - alpha * pyk;
        while (Ki.functional_proximity(xkb, skb) <= Beta) {
          alpha *= T(0.5);
          xkb = xk - alpha * pxk;
          skb = sk - alpha * pyk;
        }

        return self.template operator()<idx + 1, Start + KiRows>(alpha, px, py);
      }
    };

    do {
      /// (dakre) apply predictor correct for all cones
      const auto rd = transpose(F) * s + c;
      const auto rp = F * z - x - g;
      pred_corr_method(rd, rp);

      /// (dakre) linear combination between predictor & corrector dirs to form
      /// uniform newton step
      const auto px = px_pred + px_corr;
      const auto py = py_pred + py_corr;
      const auto pz = pz_pred + pz_corr;

      /// (dakre) backtrack alpha
      const auto alpha = backtrack(T(1), px, py);

      /// (dakre) take newton step scaled by alpha
      x = x - alpha * px;
      s = s - alpha * py;
      z = z - alpha * pz;
    } while (linalg::dot(x, s) > linalg::SolverTraits<T>::tol);

    return PrimalDualResult(linalg::SolverResult::OPTIMAL, z, x, s);
  }
};

}  // namespace convex
