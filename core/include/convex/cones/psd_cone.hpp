/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cone.hpp"
#include "linalg/operator/matrix_inverse.hpp"
#include "linalg/operator/matrix_sqrt.hpp"

namespace conic {

/**
 * @brief Circ - Jordan map/product for the PSD cone
 *
 * X circ Y = 0.5 (X Y + Y X)
 */
template <linalg::VectorContainer Vec, std::size_t N>
struct PSDCirc {
  /// @brief types and size extraction
  using T = linalg::vector_element_t<Vec>;

  /**
   * @brief circ (jordan map/product)
   *
   * X circ Y = 0.5 (X Y + Y X)
   *
   * @param x in K
   * @param y in K^*
   *
   * @return x circ y (vector)
   */
  constexpr auto operator()(const Vec& x, const Vec& y) const {
    const auto X = linalg::make_smat<N>(x);
    const auto Y = linalg::make_smat<N>(y);
    return linalg::get_svec(T(0.5) * (X * Y + Y * X));
  }
};

/** @brief psd cone class
 *
 * svec(X) in S^n_+
 */
template <linalg::VectorContainer Slack, std::size_t N>
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
   * @brief default x in K
   *
   * @return x
   */
  constexpr Slack x0_impl() const { return e_impl(); }

  /**
   * @brief default y in K^*
   *
   * @return y
   */
  constexpr Slack y0_impl() const { return e_impl(); }

  /**
   * @brief x in K check
   *
   * @param x in K
   *
   * @return T/F
   */
  constexpr bool in_cone_impl(const Slack& x) const {
    const auto X = linalg::make_smat<N, Slack>(x);
    const linalg::Cholesky chol(X);
    return chol.is_pd();
  }

  /**
   * @brief y in K^* check
   *
   * @param y in K^*
   *
   * @return T/F
   */
  constexpr bool in_dual_cone_impl(const Slack& y) const {
    return in_cone_impl(y);
  }

  /**
   * @brief alpha_max (fraction to boundary)
   *
   * sigma(alpha) = frac{1}{sup{alpha | (x + alpha p in K)}}
   *
   * X + alpha P in S^n_++
   * X + alpha P = [
   *  X^{1/2} (I + alpha X^{-1/2} P X^{-1/2}) X^{1/2}
   * ]
   * let U = X^{-1/2} P X^{-1/2} we want I + alpha U >= 0
   * I + alpha U >= 0 iff 1 + alpha lambda_i(U) >= 0 for all i
   *
   * alpha_max = cases{
   *  +inf, if lambda_i(U) >= 0 for all i
   *  -frac{1}{lambda_min(U)} else
   * }
   *
   * @param x in K
   * @param p in E
   *
   * @return alpha_max
   */
  constexpr T alpha_max_impl(const Slack& x,
                             const linalg::Vector<T, M>& p) const {
    constexpr T eta{0.99};
    if (in_cone_impl(x + p)) return eta;

    constexpr std::size_t steps{64};
    T lo{0}, hi{1};
    for (std::size_t i = 0; i < steps; ++i) {
      const T mid = T(0.5) * (lo + hi);
      if (in_cone_impl(x + mid * p)) lo = mid;
      else hi = mid;
    }
    return eta * lo;
  }

  /**
   * @brief alpha_max dual (fraction to dK^*)
   *
   * @param y in K^*
   * @param p in E^*
   *
   *
   * @return alpha_max (dual)
   */
  constexpr T alpha_max_dual_impl(const Slack& y,
                                  const linalg::Vector<T, M>& p) const {
    return alpha_max_impl(y, p);
  }

  /**
   * @brief hessian
   *
   * F(w) = -log det(w)
   * F'(w) = - w^{-1}
   * F''(w) = P(w^{-1})
   *
   * F''(w^{-1}) x = y
   *
   * @param x in K
   * @param y in K^*
   *
   * @return (H, event)
   */
  constexpr auto H_impl(const Slack& x, const Slack& y) const {
    const auto [w, event] = w_impl(x, y);
    if (linalg::is_fatal_event(event))
      return std::tuple(
          linalg::QuadraticRep<HessianMat, PSDCirc<Slack, N>>(e_impl()), event);
    const auto [w_inv, inv_event] = inv_impl(w);
    if (linalg::is_fatal_event(inv_event))
      return std::tuple(
          linalg::QuadraticRep<HessianMat, PSDCirc<Slack, N>>(e_impl()),
          inv_event);
    const auto out_event = (event != linalg::NumericalEvent::NONE ||
                            inv_event != linalg::NumericalEvent::NONE)
                               ? linalg::NumericalEvent::NUMERICAL_SAFEGUARD
                               : linalg::NumericalEvent::NONE;
    return std::tuple(
        linalg::QuadraticRep<HessianMat, PSDCirc<Slack, N>>(w_inv), out_event);
  }

  /**
   * @brief grad (jacobian)
   *
   * F'(x) = -x^{-1}
   *
   * @param x in K or y in K^*
   *
   * @return (F'(x), event)
   */
  constexpr auto grad_impl(const Slack& x) const {
    const auto [x_inv, event] = inv_impl(x);
    return std::tuple(T(-1) * x_inv, event);
  }

  /**
   * @brief barrier param
   *
   * @return nu (matrix order N, the degree of det(X))
   */
  constexpr T nu_impl() const { return static_cast<T>(N); }

  /**
   * @brief e (jordan identity)
   *
   * e = I
   *
   * @return e
   */
  constexpr auto e_impl() const {
    Slack x{};
    std::size_t k{0};
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = 0; j <= i; ++j, ++k)
        if (i == j) x(k) = T(1);
    return x;
  }

  /// Jordan square root in symmetric-vector coordinates.
  constexpr auto sqrt_impl(const Slack& x) const {
    const auto matrix = linalg::make_smat<N>(x);
    const auto [root, inv_root, event] =
        linalg::MatrixSqrt<decltype(matrix)>()(matrix);
    return std::tuple(linalg::get_svec(root), event);
  }

  /**
   * @brief w (self-scaled point)
   *
   * W = X^{1/2} (X^{1/2} Y X^{1/2})^{-1/2} X^{1/2}
   *
   * @param x in K
   * @param y in K^*
   *
   * @return (w, event)
   */
  constexpr auto w_impl(const Slack& x, const Slack& y) const {
    Slack w{};
    const auto X = linalg::make_smat<N, Slack>(x);
    const auto [X_sqrt, X_inv_sqrt, event1] =
        linalg::MatrixSqrt<decltype(X)>()(X);
    if (linalg::is_fatal_event(event1)) return std::tuple(w, event1);
    (void)X_inv_sqrt;

    const auto Y = linalg::make_smat<N, Slack>(y);
    const auto U = linalg::symmetric_part(X_sqrt * (Y * X_sqrt));
    const auto [U_sqrt, U_inv_sqrt, event2] =
        linalg::MatrixSqrt<decltype(U)>()(U);
    if (linalg::is_fatal_event(event2)) return std::tuple(w, event2);
    (void)U_sqrt;

    w = linalg::get_svec(
        linalg::symmetric_part(X_sqrt * (U_inv_sqrt * X_sqrt)));
    const auto out_event = (event1 != linalg::NumericalEvent::NONE ||
                            event2 != linalg::NumericalEvent::NONE)
                               ? linalg::NumericalEvent::NUMERICAL_SAFEGUARD
                               : linalg::NumericalEvent::NONE;
    return std::tuple(w, out_event);
  }

  /**
   * @brief circ
   *
   * x circ y
   *
   * @param x in K
   * @param y in K^*
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
   * @param x in K
   * @param y in K^*
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
   * gamma_F = -ln(det(X)) - ln(det(Y)) + nu ln(Tr(X^{1/2} Y X^{1/2}) / nu)
   *
   * @param x in K
   * @param y in K^*
   *
   * @return gamma_F
   */
  constexpr auto gamma_F_impl(const Slack& x, const Slack& y) const {
    const auto X = linalg::make_smat<N, Slack>(x);
    const auto Y = linalg::make_smat<N, Slack>(y);

    const auto [X_sqrt, X_inv_sqrt, event] =
        linalg::MatrixSqrt<decltype(X)>()(X);
    if (event != linalg::NumericalEvent::NONE) return std::tuple(T(0), event);
    (void)X_inv_sqrt;

    const auto ldl_X = linalg::LDL(X);
    const auto ldl_Y = linalg::LDL(Y);

    const auto factor_res = ldl_X.is_valid() && ldl_Y.is_valid();
    AK_ASSERT(factor_res && "Either LDL(X) or LDL(Y) is invalid");
    if (!factor_res)
      return std::tuple(T(0), linalg::NumericalEvent::FACTORIZATION_FAILED);

    const auto a = T(-1) * linalg::constexpr_ln_safe(ldl_X.det());
    const auto b = T(-1) * linalg::constexpr_ln_safe(ldl_Y.det());
    const auto tr_term = linalg::trace(X_sqrt * (Y * X_sqrt));
    const auto c = nu_impl() * linalg::constexpr_ln_safe(tr_term / nu_impl());

    return std::tuple(a + b + c, linalg::NumericalEvent::NONE);
  }

  /**
   * @brief inv
   *
   * @param x in K
   *
   * @return (inv(x), event)
   */
  constexpr auto inv_impl(const Slack& x) const {
    const auto X = linalg::make_smat<N, Slack>(x);
    const auto [X_inv, event] =
        linalg::SymmetricMatrixInverse<decltype(X)>()(X);
    return std::tuple(linalg::get_svec(linalg::symmetric_part(X_inv)), event);
  }
};

} // namespace conic
