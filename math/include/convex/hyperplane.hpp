/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set.hpp"

namespace convex {

/**
 * @brief Hyperplane defition defined on valid vector space
 *
 * H = { x | a^T x = b }
 */
template <linalg::VectorSpace Space>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Hyperplane : public ConvexSet<Space, Hyperplane<Space>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, Hyperplane<Space>>;
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

 private:
  /// @brief allow base class to invoke impl funcs
  friend ConvexBase;

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
  constexpr auto transform_impl(const AffineMap<Mat>& map) const {
    const auto aT_U =
        transpose(as_col_vector(a)) * linalg::inverse(map.linear_part());
    return Hyperplane<Space>(as_vector(aT_U),
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

  /** @brief member var in hyperplane */
  Point a;  //< n-dim vec of type T
  T b;      //< bias scalar of type T
};

}  // namespace convex
