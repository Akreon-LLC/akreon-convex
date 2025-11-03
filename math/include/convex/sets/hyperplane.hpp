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

#include "convex_set.hpp"

namespace convex {

/**
 * @brief Hyperplane defition defined on valid vector space
 *
 * H = { x | a^T x = b }
 */
template <linalg::RnSpace Space, ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Hyperplane : public ConvexSet<Space, Hyperplane<Space, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, Hyperplane<Space, Tag>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief hyperplane dimension
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief CTOR for hyperplane
   *
   * @param Point (i.e., n-dim vec of type T)
   * @param b bias scalar
   */
  constexpr explicit Hyperplane(const Point& a_in, const T& b_in)
      : a(a_in), b(b_in) {}

  /**
   * @brief contains function to see if x lies in the hyerplane defined by a^T x
   * == b
   *
   * @param point x of type T (n-dim)
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& tol) const {
    return linalg::constexpr_abs(linalg::dot(a, x) - b) < tol;
  }

  /**
   * @brief contains function to see if x lies in the hyerplane defined by a^T x
   * == b t
   *
   * When lifted to convex cone the pair (x,t) \in R^n x R_++, so last elem of
   * point x is t
   *
   * K = { (x,t) | a^T x = b t } \cup { 0 }
   *
   * @param point x of type T (n-dim)
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& tol) const {
    const auto x = std::get<0>(x_in);
    auto t = std::get<1>(x_in);
    if (t < T(0)) return false;
    t *= b;
    return linalg::constexpr_abs(linalg::dot(a, x) - t) < tol;
  }

  /**
   * @brief is well formed checks to see if the hyperplane is valid
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const {
    if (linalg::all_zero(a)) return false;
    return true;
  }

  /**
   * @brief transform implementation
   *
   * H = { x | a^T x = b }
   * f(H) = { y = f(x) | x \in H }
   * y = U x + t -> x = U^{-1} (y - t)
   * a^T U^{-1} (y - t) = b \to a^T U^{-1} y = b + a^T U^{-1}
   *
   * @param affine map
   *
   * @return transformed hyperplane
   */
  template <linalg::SquareMatrixLike Mat>
    requires(linalg::mat_cols_v<Mat> == N &&
             std::same_as<linalg::mat_type_t<Mat>, T>)
  constexpr auto transform_impl(const affine::AffineMap<Mat>& map) const {
    const auto aT_U =
        transpose(as_col_vector(a)) * linalg::inverse(map.linear_part());
    return Hyperplane<Space, Tag>(as_vector(aT_U),
                                  b + (aT_U * map.translation())(0));
  }

  /**
   * @brief is_equal implementation
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return other.self().a == a && other.self().b == b;
  }

  /**
   * @brief inverse perspective func from convex set to cone
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    if constexpr (Tag == ViewTag::ConvexSet)
      return ConeView(Hyperplane<Space, ViewTag::LiftedCone>(a, b));
    else
      return ConeView(Hyperplane<Space, Tag>(a, b));
  }

  /**
   * @brief project impl
   *
   * @param project map
   * @param point in set
   *
   * @return projected set
   */
  template <linalg::MatrixLike MatA, linalg::MatrixLike MatC>
  constexpr std::optional<linalg::Vector<T, linalg::mat_cols_v<MatA> + 1>>
  projective_map_impl(const ProjectiveMap<MatA, MatC>& map,
                      const Point& x) const {
    if (!this->contains(x)) return std::nullopt;
    return map.apply(x);
  }

 private:
  /** @brief member var in hyperplane */
  Point a;  //< n-dim vec of type T
  T b;      //< bias scalar of type T
};

}  // namespace convex
