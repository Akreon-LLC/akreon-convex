/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set.hpp"

namespace convex {

/// @brief simple enum to describe type of halfspace
enum struct HalfspaceType { CLH, OLH, CUH, OUH };

/**
 * @brief Halfspace defined on a valid vector space and one of the following
 * type traits:
 *
 * {x | a^T x <= b}... closed lower halfspace
 * {x | a^T x < b}... open lower halfspace
 * {x | a^T x >= b}... closed upper halfspace
 * {x | a^T x > b}... open upper halfspace
 */
template <linalg::VectorSpace Space>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Halfspace : public ConvexSet<Space, Halfspace<Space>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, Halfspace<Space>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief halfspace can only be defined over reals
  static_assert(!linalg::is_complex_v<T>);

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief CTOR for halfspace set
   *
   * @param Point (i.e., n-dim vector of type T)
   * @param b bias scalar
   * @param enum type describing halfspace
   */
  constexpr explicit Halfspace(const Point& a_in, const T& b_in,
                               const HalfspaceType& type_in)
      : a(a_in), b(b_in), type(type_in) {}

 private:
  /// @brief allow base class to invoke impl pure virtual (v-table like) funcs
  friend ConvexBase;

  /**
   * @brief contains function to see if x is in the halfspace a^T x <= b
   *
   * @param point x of type T with n elements
   * @param tolerance of type T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& tol) const {
    (void)tol;
    switch (type) {
      case HalfspaceType::OUH:
        return linalg::dot(a, x) > b;
      case HalfspaceType::CUH:
        return linalg::dot(a, x) >= b;
      case HalfspaceType::OLH:
        return linalg::dot(a, x) < b;
      case HalfspaceType::CLH:
      default:
        return linalg::dot(a, x) <= b;
    }
  }

  /**
   * @brief is_well_formed checks to see if underlying set is a valid convex set
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const {
    if (linalg::all_zero(a)) {
      if (type == HalfspaceType::CLH || type == HalfspaceType::OLH) {
        if (b < T(0)) return false;
      } else {
        if (b > T(0)) return false;
      }
    }
    return true;
  }

  /**
   * @brief transform halfspace by affine map
   *
   * H = { x | a^T x <= b }
   * f(H) = { y = f(x) | x \in H}... image of halfspace under f mapping
   * where affine map is: y = U x + t -> x = U^{-1} (y - t)
   * so, a^T U^{-1} (y - t) <= b \to a^T U^{-1} y <= b + a^T U^{-1} t
   * we arrive at: f(H) = { y | a^T U^{-1} y <= b + a^T U^{-1} t }
   *
   * @param affine map
   *
   * @return transformed halfspace
   */
  template <linalg::SquareMatrixLike Mat>
    requires(linalg::mat_rows_v<Mat> == N &&
             !linalg::is_complex_v<linalg::mat_type_t<Mat>>)
  constexpr auto transform_impl(const AffineMap<Mat>& map) const {
    const auto aT_U =
        transpose(as_col_vector(a)) * linalg::inverse(map.linear_part());
    return Halfspace<Space>(as_vector(aT_U), b + (aT_U * map.translation())(0),
                            type);
  }

  /**
   * @brief is_equal implementation
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(
      const ConvexSet<Space, Halfspace<Space>>& other) const {
    return other.self().a == a && other.self().b == b &&
           other.self().type == type;
  }

  /** @brief member variables in the halfspace */
  Point a;             //< n-dim vector of type T
  T b;                 //< bias scalar of type T
  HalfspaceType type;  //< holds value of halfspace enum type
};

}  // namespace convex
