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

namespace conic {

/**
 * @brief soc cone class
 *
 * K_soc = { (x, t) | ||x|| <= t }
 */
template <linalg::VectorLike Slack>
struct SecondOrderCone
    : public Cone<Slack, SecondOrderCone<Slack>, ConeType::SYMMETRIC> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Slack, SecondOrderCone<Slack>, ConeType::SYMMETRIC>;

  /// @brief helpers to get T and M
  using ConeBase::M;
  using typename ConeBase::T;

  /**
   * @brief default interior point
   *
   * @return vector
   */
  constexpr Slack x0_impl() const {
    Slack s{};
    s(M - 1) = T(1);
    return s;
  }

  /**
   * @brief in interior of K
   *
   * @param slack vector
   *
   * @return bool
   */
  constexpr bool in_cone_impl(const Slack& x) const {
    T res{};
    for (size_t i = 0; i < M - 1; ++i) res += x(i) * x(i);
    res = linalg::constexpr_sqrt(res);
    return res < x(M - 1);
  }

  /**
   * @brief fraction to boundary dK
   *
   * sigma_x = \frac{1}{sup{alpha | x + alpha p in K}}
   * p = [ delta x; delta t ]
   * ||x + alpha delta(x)||_2^2 <= (t + alpha delta(t))^2
   * 0 <= (t + alpha delta(t))^2 - ||x + alpha delta(x)||_2^2
   * -inf(t^2 + 2 alpha t delta(t) + alpha^2 delta(t)^2 - x^2 - 2 alpha x
   * delta(x) - alpha^2 delta(x)^2)
   *
   * a alpha^2 + b alpha + c >= 0
   *
   * a = delta(t)^2 - delta(x)^2
   * b = 2 (t delta(t) - x^T delta(x))
   * c = t^2 - x^2 > 0
   *
   * alpha_max = piecewise{
   *  \frac{-b +- \sqrt(b^2 - 4 a c)}{2a}, a > 0,
   *  \frac{-c}{b}, a = 0, b > 0,
   *  +inf, a < 0
   * }
   *
   * @param x \in K
   * @param p \in E (delta dir)
   *
   * @return \alpha_max
   */
  constexpr T alpha_max_impl(const Slack& x,
                             const linalg::Vector<T, M>& p) const {
    constexpr T eta{0.99};
    constexpr T eps{1e-8};
    T alpha{std::numeric_limits<T>::infinity()};
    T alpha_t{std::numeric_limits<T>::infinity()};

    const auto var = std::get<0>(linalg::splice<M - 1>(x));
    const auto t = x(M - 1);
    const auto delta_var = std::get<0>(linalg::splice<M - 1>(p));
    const auto delta_t = p(M - 1);

    const auto a = delta_t * delta_t - linalg::dot(delta_var, delta_var);
    const auto b = T(2) * (t * delta_t - linalg::dot(var, delta_var));
    const auto c = t * t - linalg::dot(var, var);

    if (linalg::constexpr_abs(a) <= eps)
      alpha = b < T(0) ? -c / b : alpha;
    else {
      const auto disc = b * b - T(4) * a * c;
      if (disc > T(0)) {
        const auto s = linalg::constexpr_sqrt(disc);
        const auto r1 = (T(-1) * b - s) / (T(2) * a);
        const auto r2 = (T(-1) * b + s) / (T(2) * a);

        if (r1 > T(0)) alpha = std::min(alpha, r1);
        if (r2 > T(0)) alpha = std::min(alpha, r2);
      }
    }

    /// (dakre) ensure t + alpha dt > 0
    if (delta_t < T(0)) alpha_t = -t / delta_t;

    return eta * std::min(alpha, alpha_t);
  }

  /**
   * @brief hessian
   *
   * F''(w) -> pushforward op same style as R^n_+
   *
   * @param x \in K
   * @param s \in K^*
   *
   * @return H
   */
  constexpr auto H_impl(const Slack& x, const Slack& s) const {
    (void)x, (void)s;
    /// TODO - impl
    return false;
  }

  /**
   * @brief hessian inverse
   *
   * F''(w)^{-1} -> pullback operator same style as R^n_+
   *
   * @param x \in K
   * @param s \in K^*
   *
   * @return H^{-1}
   */
  constexpr auto H_inv_impl(const Slack& x, const Slack& s) const {
    (void)x, (void)s;
    /// TODO - impl
    return false;
  }

  /**
   * @brief barrier param
   *
   * K = {(x,t) | t >= ||x||}
   * F(x,t) = -log(t^2 - ||x||^2)
   * t^2 - ||x||^2 -> lam1 = t + ||x||, lam2 = t - ||x||
   * det(x) = lam1*lam2 -> F = -log lam1 - log lam2
   *
   * ^ 2 log terms -> 2 barrier params
   *
   * @return nu
   */
  constexpr T nu_impl() const { return T(2); }

  /**
   * @brief circ (jordan map/product)
   *
   * x \circ s = (x,t1) \circ (y,t2) = (x t2 + y t1, x^T y + t1 t2)
   *
   * @param x \in K
   * @param s \in K^*
   *
   * @return x \circ s (bilinear product)
   */
  constexpr auto circ_impl(const Slack& x, const Slack& s) const {
    const auto primal = std::get<0>(linalg::splice<M - 1>(x));
    const auto t1 = x(M - 1);
    const auto dual = std::get<0>(linalg::splice<M - 1>(s));
    const auto t2 = s(M - 1);
    const auto a = t2 * primal + t1 * dual;
    const auto b = linalg::dot(primal, dual) + t1 * t2;

    Slack res{};
    for (size_t i = 0; i < M - 1; ++i) res(i) = a(i);
    res(M - 1) = b;
    return res;
  }

  /**
   * @brief e (jordan identity)
   *
   * e = (0,1)
   * (x,t) \circ e = (x,t)
   *
   * @return e
   */
  constexpr auto e_impl() const {
    Slack e{};
    e(M - 1) = T(1);
    return e;
  }
};

}  // namespace conic
