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

#include "linalg/linalg.hpp"

namespace lti {

enum Stability : int { Unstable = 0, Asymptotic, Marginal };

template <typename ScalarT, size_t N, size_t P, size_t Q,
          typename DerivedSystem>
  requires(linalg::FieldLike<ScalarT>)
class LTISystem {
 public:
  constexpr explicit LTISystem(const linalg::Matrix<ScalarT, N, N>& A_in,
                               const linalg::Matrix<ScalarT, N, P>& B_in,
                               const linalg::Matrix<ScalarT, Q, N>& C_in,
                               const linalg::Matrix<ScalarT, Q, P>& D_in)
      : A(A_in), B(B_in), C(C_in), D(D_in) {}

  LTISystem(const LTISystem& system) = delete;
  LTISystem(LTISystem&& system) = delete;
  LTISystem& operator=(const LTISystem& system) = delete;
  LTISystem& operator=(LTISystem&& system) = delete;

  constexpr auto step(
      const linalg::Vector<ScalarT, N>& x,
      linalg::Vector<ScalarT, P> u = linalg::Vector<ScalarT, P>{}) const {
    return self().step_impl(x, u);
  }

  constexpr auto output(
      const linalg::Vector<ScalarT, N>& x,
      linalg::Vector<ScalarT, P> u = linalg::Vector<ScalarT, P>{}) const {
    return self().output_impl(x, u);
  }

  constexpr int is_stable(ScalarT tol = 1e-8) const {
    return self().is_stable_impl(tol);
  }

  constexpr bool is_reachable() const { return self().is_reachable_impl(); }

  constexpr bool is_observable() const { return self().is_observable_impl(); }

 protected:
  linalg::Matrix<ScalarT, N, N> A;
  linalg::Matrix<ScalarT, N, P> B;
  linalg::Matrix<ScalarT, Q, N> C;
  linalg::Matrix<ScalarT, Q, P> D;

 private:
  constexpr const DerivedSystem& self() const {
    return static_cast<const DerivedSystem&>(*this);
  }
};

}  // namespace lti
