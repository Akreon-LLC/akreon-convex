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

#include "linalg/linalg_util.hpp"

namespace convex {

/**
 * @brief No equality-coupling case for empty A.
 */
template <linalg::VectorLike Var, linalg::VectorLike NuVar>
struct NoEqualityCoupling {
  /**
   * @brief solve for pz step
   *
   * @param M^{-1} (LDL(M))
   * @param pz_rhs (rhs to be solved)
   * @param rd (dual residual)
   * @param req (equality residual - unused)
   *
   * @return (pz, pnu, event)
   */
  constexpr auto solve_step(const auto& M_inv, const Var& pz_rhs, const Var& rd,
                            const NuVar& /* req */) const {
    const auto [pz, event] = M_inv.solve(pz_rhs + rd);
    return std::tuple(pz, NuVar{}, event);
  }
};

/**
 * @brief Equality-coupling wrapper for Schur-complement based solve.
 *
 * 2x2 saddle system
 * [ M A^T ] [ pz; pnu] = [ hat{r}_d; req ]
 * [ A 0   ]
 *
 * M block
 * M = F^T H F
 *
 * Schur complement
 * S = A M^{-1} A^T
 */
template <linalg::MatrixLike EqMat, typename SchurFactor,
          linalg::VectorLike Var, linalg::VectorLike NuVar>
struct SchurEqualityCoupling {
  /**
   * @brief CTOR
   *
   * @param A (eq mat)
   * @param S_inv (schur factor)
   */
  constexpr explicit SchurEqualityCoupling(const EqMat& A_in,
                                           const SchurFactor& S_inv_in)
      : A(A_in), S_inv(S_inv_in) {}

  /**
   * @brief solve for pz step
   *
   * pnu = S^{-1}(A M^{-1} hat{r}_d - req)
   * pz = M^{-1}(hat{r}_d - A^T pnu)
   * where hat{r}_d = pz_rhs + rd
   *
   * @param M^{-1} (LDL(M))
   * @param pz_rhs (rhs to be solved)
   * @param rd (dual residual)
   * @param req (equality residual)
   *
   * @return (pz, pnu, event)
   */
  constexpr auto solve_step(const auto& M_inv, const Var& pz_rhs, const Var& rd,
                            const NuVar& req) const {
    const auto [M_inv_pz, event_one] = M_inv.solve(pz_rhs + rd);
    if (linalg::is_fatal_event(event_one))
      return std::tuple(Var{}, NuVar{}, event_one);

    const auto schur_rhs = A * M_inv_pz - req;
    const auto [pnu, event_two] = S_inv.solve(schur_rhs);
    if (linalg::is_fatal_event(event_two))
      return std::tuple(Var{}, NuVar{}, event_two);

    const auto [pz, event_three] =
        M_inv.solve(pz_rhs + rd - linalg::transpose(A) * pnu);
    if (linalg::is_fatal_event(event_three))
      return std::tuple(Var{}, NuVar{}, event_three);

    return std::tuple(pz, pnu, linalg::NumericalEvent::NONE);
  }

  /** @brief member vars **/
  EqMat A{};         ///< Equality mat
  SchurFactor S_inv; ///< Schur factor S = (A M^{-1} A^T)^{-1}
};

/**
 * @brief Scalar Schur complement wrapper for 1x1 equality case.
 */
template <linalg::MatrixLike EqMat, linalg::FieldLike Scalar,
          linalg::VectorLike Var, linalg::VectorLike NuVar>
struct ScalarEqualityCoupling {
  /**
   * @brief CTOR
   *
   * @param A (equality matrix)
   * @param s (scalar)
   */
  constexpr explicit ScalarEqualityCoupling(const EqMat& A_in,
                                            const Scalar& s_in)
      : A(A_in), s(s_in) {}

  /**
   * @brief solve for pz step
   *
   * pnu = S^{-1}(A M^{-1} hat{r}_d - req)
   * pz = M^{-1}(hat{r}_d - A^T pnu)
   * where hat{r}_d = pz_rhs + rd
   *
   * @param M^{-1} (LDL(M))
   * @param pz_rhs (rhs to be solved)
   * @param rd (dual residual)
   * @param req (equality residual)
   *
   * @return (pz, pnu, event)
   */
  constexpr auto solve_step(const auto& M_inv, const Var& pz_rhs, const Var& rd,
                            const NuVar& req) const {
    const auto [M_inv_pz, event_one] = M_inv.solve(pz_rhs + rd);
    if (linalg::is_fatal_event(event_one))
      return std::tuple(Var{}, NuVar{}, event_one);

    const auto schur_rhs = A * M_inv_pz - req;
    NuVar pnu{};
    pnu(0) = schur_rhs(0) / s;

    const auto [pz, event_two] =
        M_inv.solve(pz_rhs + rd - linalg::transpose(A) * pnu);
    if (linalg::is_fatal_event(event_two))
      return std::tuple(Var{}, NuVar{}, event_two);

    return std::tuple(pz, pnu, linalg::NumericalEvent::NONE);
  }

  /** @brief member vars **/
  EqMat A{};  ///< Equality matrix
  Scalar s{}; ///< scalar for S
};

} // namespace convex
