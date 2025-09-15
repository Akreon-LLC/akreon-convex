/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "lti_system.hpp"

namespace lti {

template <typename ScalarT, size_t N, size_t P, size_t Q>
class LTIDiscreteSystem
    : public LTISystem<ScalarT, N, P, Q, LTIDiscreteSystem<ScalarT, N, P, Q>> {
 public:
  constexpr explicit LTIDiscreteSystem(
      const linalg::Matrix<ScalarT, N, N>& A_in,
      const linalg::Matrix<ScalarT, N, P>& B_in,
      const linalg::Matrix<ScalarT, Q, N>& C_in,
      linalg::Matrix<ScalarT, Q, P> D_in = linalg::Matrix<ScalarT, Q, P>{})
      : LTISystem<ScalarT, N, P, Q, LTIDiscreteSystem<ScalarT, N, P, Q>>(
            A_in, B_in, C_in, D_in) {}

  constexpr auto step_impl(const linalg::Vector<ScalarT, N>& x,
                           const linalg::Vector<ScalarT, P>& u) const {
    return this->A * x + this->B * u;
  }

  constexpr auto output_impl(const linalg::Vector<ScalarT, N>& x,
                             const linalg::Vector<ScalarT, P>& u) const {
    return this->C * x + this->D * u;
  }

  constexpr int is_stable_impl(const ScalarT& tol) const {
    const auto eig_decomp = linalg::EigenDecomposition(this->A);
    const auto eigs = eig_decomp.get_eigenvalues();
    bool marginal = false;
    for (size_t i = 0; i < N; ++i) {
      const auto re = eigs(i).real();
      const auto im = eigs(i).imag();
      const auto modulus = linalg::constexpr_sqrt(re * re + im * im);
      // (dakre) TODO scrutinize this a little more... tol 1e-8 is okay for now
      // though
      if (modulus > ScalarT(1) + tol) return Stability::Unstable;
      if (linalg::constexpr_abs(modulus - ScalarT(1)) < tol) marginal = true;
    }
    return marginal ? Stability::Marginal : Stability::Asymptotic;
  }

 private:
};

}  // namespace lti
