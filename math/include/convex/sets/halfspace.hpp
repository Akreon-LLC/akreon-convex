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
template <linalg::VectorSpace Space, HalfspaceType Type>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Halfspace : public ConvexSet<Space, Halfspace<Space, Type>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, Halfspace<Space, Type>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief CTOR for halfspace set
   *
   * @param Point (i.e., n-dim vector of type T)
   * @param b bias scalar
   */
  constexpr explicit Halfspace(const Point& a_in, const T& b_in)
      : a(a_in), b(b_in) {}

  /**
   * @brief CTOR for halfspace in conic view
   *
   * @param Point (n-dim vec)
   * @param b bias scalar
   * @param t cone view scalar
   */
  constexpr explicit Halfspace(const Point& a_in, const T& b_in, const T& t_in)
      : a(a_in), b(b_in), t(t_in) {}

 protected:
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
  constexpr bool contains_impl(const Point& x, const T& /*tol*/) const {
    T alpha(b);
    if (t.has_value()) alpha *= t.value();

    switch (Type) {
      case HalfspaceType::OUH:
        return linalg::dot(a, x) > alpha;
      case HalfspaceType::CUH:
        return linalg::dot(a, x) >= alpha;
      case HalfspaceType::OLH:
        return linalg::dot(a, x) < alpha;
      case HalfspaceType::CLH:
      default:
        return linalg::dot(a, x) <= alpha;
    }
  }

  /**
   * @brief is_well_formed checks to see if underlying set is a valid convex set
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const {
    if (t.has_value())
      if (t <= T(0)) return false;

    if (linalg::all_zero(a)) {
      if constexpr (Type == HalfspaceType::CLH || Type == HalfspaceType::OLH) {
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
    requires(std::same_as<T, linalg::mat_type_t<Mat>> &&
             linalg::mat_cols_v<Mat> == N)
  constexpr auto transform_impl(const AffineMap<Mat>& map) const {
    if (t.has_value()) return *this;

    const auto aT_U =
        transpose(as_col_vector(a)) * linalg::inverse(map.linear_part());
    return Halfspace<Space, Type>(as_vector(aT_U),
                                  b + (aT_U * map.translation())(0));
  }

  /**
   * @brief is_equal implementation
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return other.self().a == a && other.self().b == b && other.self().t == t;
  }

  /**
   * @brief inverse perspective func from convex set to cone
   *
   * @param t cone slice
   *
   * @return cone
   */
  constexpr auto to_conic_view_impl(const T& t_in) const {
    return Halfspace<Space, Type>(a, b, t_in);
  }

  /**
   * @brief is cone
   *
   * @return T/F
   */
  constexpr bool is_conic_view_impl() const { return t.has_value(); }

  /** @brief member variables in the halfspace */
  Point a;             //< n-dim vector of type T
  T b;                 //< bias scalar of type T
  std::optional<T> t;  //< conic view t scalar
};

}  // namespace convex
