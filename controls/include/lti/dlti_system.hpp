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
    bool marginal = false;
    const auto eig_decomp = linalg::EigenDecomposition(this->A);
    const auto lambda_A = eig_decomp.get_eigenvalues();
    for (size_t i = 0; i < N; ++i) {
      const auto re = lambda_A(i).real();
      const auto im = lambda_A(i).imag();
      const auto modulus = linalg::constexpr_sqrt(re * re + im * im);
      // (dakre) TODO scrutinize this a little more... tol 1e-8 is okay for now
      // though
      if (modulus > ScalarT(1) + tol) return Stability::Unstable;
      if (linalg::constexpr_abs(modulus - ScalarT(1)) < tol) marginal = true;
    }
    return marginal ? Stability::Marginal : Stability::Asymptotic;
  }

  constexpr bool is_reachable() const {
    // (dakre) there's two ways to do this:
    // then ss is reachable
    // 1. form cayley hamilton Wr = [B AB A^2B ...], rank(Wr) == n ? yes : no
    // 2. PBH test: rank([\lambda * I - A B]) == n ? yes : no
    // The numerical properties are better suited for PBH (below)
    const auto eig_decomp = linalg::EigenDecomposition(this->A);
    const auto lambda_A = eig_decomp.get_eigenvalues();

    using ST = std::conditional_t<linalg::is_complex_v<ScalarT>, ScalarT,
                                  std::complex<linalg::scalar_type_t<ScalarT>>>;

    const auto I = linalg::make_identity<ST, N>();
    linalg::Matrix<ST, N, N + P> M;
    for (size_t i = 0; i < N; ++i) {
      const auto lambda_I = lambda_A(i) * I - as_complex_matrix(this->A);
      M = linalg::set_block<0, N, 0, N>(M, lambda_I);
      if constexpr (P > 0)
        M = linalg::set_block<0, N, N, N + P>(M, as_complex_matrix(this->B));
      const auto qr_M = linalg::QR(linalg::transpose(M));
      if (qr_M.get_rank() != N) return false;
    }
    return true;
  }

 private:
};

}  // namespace lti
