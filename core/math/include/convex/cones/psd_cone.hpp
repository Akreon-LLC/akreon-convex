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

#include "cone.hpp"
#include "linalg/operator/matrix_inverse.hpp"
#include "linalg/operator/matrix_sqrt.hpp"

namespace conic {

/**
 * @brief Circ - Jordan map/product for the PSD cone
 *
 * X \circ Y = 0.5 (X Y + Y X)
 */
template <linalg::VectorLike Vec, size_t N>
struct PSDCirc {
  /// @brief types and size extraction
  using T = linalg::vec_type_t<Vec>;

  /**
   * @brief circ (jordan map/product)
   *
   * X \circ Y = 0.5 (X Y + Y X)
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return x \circ y (vector)
   */
  constexpr auto operator()(const Vec& x, const Vec& y) const {
    const auto X = linalg::make_smat<N, Vec>(x);
    const auto Y = linalg::make_smat<N, Vec>(y);
    return linalg::get_lower_triangle_with_diag(T(0.5) * (X * Y + Y * X));
  }
};

/** @brief psd cone class
 *
 * svec(X) \in S^n_+
 */
template <linalg::VectorLike Slack, size_t N>
struct PositiveSemidefiniteCone
    : public Cone<Slack, PositiveSemidefiniteCone<Slack, N>,
                  ConeType::SYMMETRIC> {
  /// @brief bring into scope alias from cone base
  using ConeBase =
      Cone<Slack, PositiveSemidefiniteCone<Slack, N>, ConeType::SYMMETRIC>;

  /// @brief helpers to get T and M
  using ConeBase::eps;
  using ConeBase::M;
  using typename ConeBase::HessianMat;
  using typename ConeBase::T;

  static_assert(M == N * (N + 1) / 2,
                "Encoded dim for PSD Cone does not align with slack dim");

  /**
   * @brief default x \in K
   *
   * @return x
   */
  constexpr Slack x0_impl() const { return e_impl(); }

  /**
   * @brief default y \in K^*
   *
   * @return y
   */
  constexpr Slack y0_impl() const { return e_impl(); }

  /**
   * @brief x \in K check
   *
   * @param x \in K
   *
   * @return T/F
   */
  constexpr bool in_cone_impl(const Slack& x) const {
    const auto X = linalg::make_smat<N, Slack>(x);
    const linalg::LDL ldl(X);
    return ldl.is_psd();
  }

  /**
   * @brief y \in K^* check
   *
   * @param y \in K^*
   *
   * @return T/F
   */
  constexpr bool in_dual_cone_impl(const Slack& y) const {
    return in_cone_impl(y);
  }

  /**
   * @brief alpha_max (fraction to boundary)
   *
   * TODO - impl
   *
   * @param x \in K
   * @param p \in E
   *
   * @return \alpha_max
   */
  constexpr T alpha_max_impl(const Slack& /*x*/,
                             const linalg::Vector<T, M>& /*p*/) const {
    constexpr T eta{0.99};
    return eta;
  }

  /**
   * @brief alpha_max dual (fraction to dK^*)
   *
   * @param y \in K^*
   * @param p \in E^*
   *
   *
   * @return \alpha_max (dual)
   */
  constexpr T alpha_max_dual_impl(const Slack& y,
                                  const linalg::Vector<T, M>& p) const {
    return alpha_max_dual_impl(y, p);
  }

  /**
   * @brief hessian
   *
   * TODO - impl
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return H
   */
  constexpr auto H_impl(const Slack& /* x */, const Slack& /* y */) const {
    return HessianMat{};
  }

  /**
   * @brief grad (jacobian)
   *
   * TODO - impl
   *
   * @param x \in K or y \in K^*
   *
   * @return F'(x)
   */
  constexpr auto grad_impl(const Slack& /* x */) const { return Slack{}; }

  /**
   * @brief barrier param
   *
   * @return nu (M)
   */
  constexpr T nu_impl() const { return static_cast<T>(M); }

  /**
   * @brief e (jordan identity)
   *
   * e = I
   *
   * @return e
   */
  constexpr auto e_impl() const {
    Slack x{};
    size_t k{0};
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j <= i; ++j, ++k)
        if (i == j) x(k) = T(1);
    return x;
  }

  /**
   * @brief w (self-scaled point)
   *
   * TODO - impl
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return w
   */
  constexpr auto w_impl(const Slack& /* x */, const Slack& /* y */) const {
    return Slack{};
  }

  /**
   * @brief circ
   *
   * x \circ y
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return x circ y
   */
  constexpr auto circ_impl(const Slack& x, const Slack& y) const {
    return PSDCirc<Slack, N>()(x, y);
  }

  /**
   * @brief trace
   *
   * tr(XY)
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return tr(XY)
   */
  constexpr T trace_impl(const Slack& x, const Slack& y) const {
    const auto X = linalg::make_smat<N, Slack>(x);
    const auto Y = linalg::make_smat<N, Slack>(y);
    return linalg::trace(X * Y);
  }

  /**
   * @brief gamma_F
   *
   * TODO - impl
   * \gamma_F = -ln(det(X)) - ln(det(Y)) + nu ln(Tr(X^{1/2} Y X^{1/2}))
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return gamma_F
   */
  constexpr T gamma_F_impl(const Slack& /* x */, const Slack& /* y */) const {
    return T(0);
  }

  /**
   * @brief inv
   *
   * @param x \in K
   *
   * @return inv(x)
   */
  constexpr auto inv_impl(const Slack& x) const {
    const auto X = linalg::make_smat<N, Slack>(x);
    const auto X_inv = linalg::SymmetricMatrixInverse<decltype(X)>()(X);
    return linalg::get_lower_triangle_with_diag(X_inv);
  }

  /**
   * @brief sqrt
   *
   * X^{1/2}
   *
   * @param x \in K
   *
   * @return sqrt(x)
   */
  constexpr auto sqrt(const Slack& x) const {
    const auto X = linalg::make_smat<N, Slack>(x);
    const auto X_sqrt = linalg::MatrixSqrt<decltype(X)>()(X);
    return linalg::get_lower_triangle_with_diag(X_sqrt);
  }
};

}  // namespace conic
