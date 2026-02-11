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
 * @brief Primal Dual Interior Point Method Solver
 */
template <BuilderLike Builder>
struct PrimalDualIPM {
  /// @brief alias and helpers to get types & sizes
  using DecVar = builder_dom_t<Builder>;
  using T = linalg::space_type_t<DecVar>;
  using N = linalg::space_dim_v<DecVar>;
  using Schur = linalg::Matrix<T, N, N>;
  using Var = linalg::Vector<T, N>;

  /**
   * @brief solve
   *
   * @param prob ConvexProblem
   *
   * @return result
   */
  constexpr auto solve(
      const ConvexProblem<Builder, StageType::PROBLEM_SET>& prob) const {
    if (!prob.validate()) return false;

    const auto F = prob.constraint_map().linear_part();
    const auto K = prob.constraint_cones();

    static constexpr size_t M = linalg::mat_rows_v<decltype(F)>;
    static constexpr size_t NumCones = std::tuple_size<decltype(K)>::value;

    linalg::Vector<T, M> x{}, s{}, px_pred{}, px_corr{}, py_pred{}, py_corr{};
    Var z{}, pz_rhs_pred{}, pz_rhs_corr{}, pz_pred{}, pz_corr{};
    Schur S{};
    /**
     * @brief process cones for predictor corrector method
     *
     * preorder
     * (1) compute hessian at current iterate
     * (2) preorder accumuate schur (S) + rhs for pz calculation
     * postorder
     * (3) only once factor schur (S) and compute pz (global \in Rn)
     * (4) for each cone solve qy = -py & px
     */
    auto pred_corr_method = [&]<size_t idx = 0, size_t Start = 0>() {
      if constexpr (idx >= NumCones)
        return false;
      else {
        const auto Ki = std::get<idx>(K);
        using Slack = cone_slack_t<decltype(Ki)>;
        constexpr size_t KiRows = linalg::vec_size_v<Slack>;
        constexpr size_t End = Start + KiRows;

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
        const AugmentedKKT<decltype(Hes.H), decltype(F_blk)> kkt;
        const auto Si = kkt.make_schur(Hes, F_blk);
        S += Si;

        const auto FT = linalg::transpose(F_blk);
        const auto rc_pred = sk;
        const auto rc_corr = sk + Ki.mu(xk, sk) * Ki.grad(xk);

        pz_rhs_pred += FT* rc_pred pz_rhs_corr += FT * rc_corr;

        const auto factored = pred_corr_method<idx + 1, Start + KiRows>();

        if (!factored) {
          const auto ldl = LDL(S);
          pz_pred = ldl.solve(pz_rhs_pred);
          pz_corr = ldl.solve(pz_rhs_corr);
        }

        const auto py_pred_tmp = Hes.apply(F * pz_pred) - rc_pred;
        const auto py_corr_tmp = Hes.apply(F * pz_corr) - rc_corr;
        const auto px_pred_tmp = F * pz_pred;
        const auto px_corr_tmp = F * pz_corr;

        for (size_t i = Start, j = 0; i < End; ++i, ++j) {
          px_pred(i) = px_pred_tmp(j);
          px_corr(i) = px_corr_tmp(j);
          py_pred(i) = py_pred_tmp(j);
          py_corr(i) = py_corr_tmp(j);
        }

        return true;
      }
    };

    do {
      /// (dakre) apply predictor correct for all cones
      pred_corr_method();

      /// (dakre) linear combination between predictor & corrector dirs to form
      /// uniform newton step
      const auto px = px_pred + px_corr;
      const auto py = py_pred + py_corr;
      const auto pz = pz_pred + pz_corr;

      /// (dakre) TODO backtrack alpha on functional proximity of each cone
      /// applying newton step to next iterate
    } while (linalg::dot(x, s) > SolverTraits<T>::tol);

    return true;
  }
};

}  // namespace convex
