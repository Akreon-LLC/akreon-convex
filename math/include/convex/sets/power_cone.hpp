/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set.hpp"

namespace convex {
/**
 * @brief power cone
 *
 * { (x, y) \in R^n_+ x R | |y| <= x_1^\alpha_1 * x_2^\alpha_2 * ... , \sum_i^n
 * \alpha_i <= 1, \forall \alpha >= 0 }
 */
template <linalg::RnSpace Space, ViewTag Tag = ViewTag::ConvexCone>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class PowerCone : public ConvexSet<Space, PowerCone<Space, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, PowerCone<Space, Tag>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief default power cone CTOR
   */
  constexpr explicit PowerCone() {
    for (size_t i = 0; i < N; ++i) roots[i] = N;
  }

  /**
   * @brief power cone (roots provided)
   *
   * @param roots \in N
   */
  constexpr explicit PowerCone(const linalg::Vector<size_t, N>& roots_in)
      : roots(roots_in) {}

  /**
   * @brief power cone contains \subset R^n_+ x R
   *
   * @param x \in R^n_+ x R
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& tol) const {
    return contains_impl(std::tuple(x, T(1)), tol);
  }

  /**
   * @brief power cone contains \subset R^n_+ x R
   *
   * @param (x, t) \in R^n_+ x R
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& /*tol*/) const {
    const auto x = std::get<0>(x_in);
    const auto t = std::get<1>(x_in);

    T prod(1);
    for (size_t i = 0; i < N; ++i)
      prod *= linalg::constexpr_nrt<T>(x(i), roots(i));

    return prod >= t;
  }

  /**
   * @brief is well formed check
   *
   * @param tolerance scalar
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl(T tol = 1e-8) const {
    T sum(0);
    std::for_each(roots.begin(), roots.end(),
                  [&](const auto& root) { sum += T(1) / T(root); });
    return linalg::constexpr_abs(sum - T(1)) < tol;
  }

  /**
   * @brief is equal impl
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    for (size_t i = 0; i < N; ++i)
      if (other.self().roots(i) != roots(i)) return false;
    return true;
  }

  /**
   * @brief transform impl
   *
   * @return self
   */
  template <linalg::MatrixLike Mat>
  constexpr auto transform_impl(const AffineMap<Mat>& /* map */) const {
    return *this;
  }

  /**
   * @brief cone view impl
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    return ConeView(PowerCone<Space, ViewTag::ConvexCone>(roots));
  }

  /**
   * @brief project impl
   *
   * @param project map
   * @param in set
   *
   * @return projected set
   */
  template <linalg::MatrixLike MatA, linalg::MatrixLike MatC>
    requires(Tag == ViewTag::ConvexSet)
  constexpr std::optional<linalg::Vector<T, linalg::mat_cols_v<MatA> + 1>>
  project_impl(const ProjectiveMap<MatA, MatC>& /* map */,
               const Point& /* x */) const {
    return std::nullopt;
  }

 private:
  /** @brief member vars */
  linalg::Vector<size_t, N> roots;  //< alphas encoded as roots
};
}  // namespace convex
