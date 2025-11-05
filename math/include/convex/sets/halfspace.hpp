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
 * @brief Halfspace defined on a valid vector space and one of the following
 * type traits:
 *
 * {x | a^T x <= b}... closed lower halfspace
 * {x | a^T x < b}... open lower halfspace
 * {x | a^T x >= b}... closed upper halfspace
 * {x | a^T x > b}... open upper halfspace
 */
template <linalg::RnSpace Space, HalfspaceType Type,
          ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Halfspace : public ConvexSet<Space, Halfspace<Space, Type, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, Halfspace<Space, Type, Tag>>;
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
   * @brief contains function to see if x is in the halfspace a^T x <= b
   *
   * @param point x of type T with n elements
   * @param tolerance of type T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& /*tol*/) const {
    switch (Type) {
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
   * @brief contains function to see if x is in the lifted halfspace
   *
   * Standard inverse perspective function:
   * (1) C' = { (x,t) \in R^n x R_++ | a^T x <= b t }
   *
   * Epi / Hypo -> Lifted Halfspace
   * (2) Epi: C' = { (x,t) \in R^n x R | a^T x <= t - b }
   * (3) Hypo: C' = { (x,t) \in R^n x R | t <= a^T x + b }
   * { (x, t) \in R^n x R | -a^T x <= b - t}
   *
   * @param (x, t) \in R^n x R_++ or \in R^n x R depending on lift
   * @param tolerance of type T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& /*tol*/) const {
    const auto x = std::get<0>(x_in);
    auto t = std::get<1>(x_in);

    if constexpr (Tag == ViewTag::EpigraphCone) {
      t -= b;
    } else if constexpr (Tag == ViewTag::HypographCone) {
      t = b - t;
    } else {
      if (t < T(0)) return false;
      t *= b;
    }

    switch (Type) {
      case HalfspaceType::OUH:
        return linalg::dot(a, x) > t;
      case HalfspaceType::CUH:
        return linalg::dot(a, x) >= t;
      case HalfspaceType::OLH:
        return linalg::dot(a, x) < t;
      case HalfspaceType::CLH:
      default:
        return linalg::dot(a, x) <= t;
    }
  }

  /**
   * @brief is_well_formed checks to see if underlying set is a valid convex
   * set
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const {
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
  constexpr auto transform_impl(const affine::AffineMap<Mat>& map) const {
    const auto aT_U =
        transpose(as_col_vector(a)) * linalg::inverse(map.linear_part());
    return Halfspace<Space, Type, Tag>(as_vector(aT_U),
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
      return ConeView(Halfspace<Space, Type, ViewTag::LiftedCone>(a, b));
    else if constexpr (Tag == ViewTag::HypographCone)
      return ConeView(
          Halfspace<Space, Type, ViewTag::HypographCone>(T(-1) * a, b));
    else
      return ConeView(Halfspace<Space, Type, Tag>(a, b));
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
  /** @brief member variables in the halfspace */
  Point a;  //< n-dim vector of type T
  T b;      //< bias scalar of type T
};

}  // namespace convex
