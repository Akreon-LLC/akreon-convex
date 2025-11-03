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
 * @brief Norm cone definition valid on a vector space dim(R^n + 1)
 *
 * { (x, t) | ||x|| <= t }... closed norm cone defined on L1, L2, Linf
 * { (x, t) | ||x|| < t }... open cone defined on L1, L2, Linf
 */
template <linalg::RnSpace Space, NormConeType Type,
          ViewTag Tag = ViewTag::LiftedCone>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class NormCone : public ConvexSet<Space, NormCone<Space, Type, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, NormCone<Space, Type, Tag>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief CTOR for norm cone
   */
  constexpr explicit NormCone() {}

  /**
   * @brief CTOR for norm cone
   *
   * Transformed from norm ball or rotated SOC
   *
   * @param center point from norm ball
   * @param r (radius from norm ball)
   */
  constexpr explicit NormCone(const Point& c_in, const T& r_in)
      : c(c_in), r(r_in) {}

  /**
   * @brief contains function to see if x is in the norm cone (y, t) \in K
   *
   * K = {(x, t) \in R^n x R | ||x|| <= 1}
   *
   * @param point (x,t) - lifted by default
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& tol) const {
    return contains_impl(std::make_tuple(x, T(1)), tol);
  }

  /**
   * @brief contains function to see if x is in the norm cone (y, t) \in K
   *
   * (1) Standard Cone: K = {(x, t) \in R^n x R_+ | ||x|| <= t}
   * (2) Epigraph Cone: K = {(x, t) \in R^n x R | ||x|| <= t}
   * (3) Hypograph Cone: K = {(x, t) \in R^n x R | ||x|| <= -t}
   *
   * @param point (x,t) - lifted by default
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& /* tol */) const {
    const auto x = std::get<0>(x_in);
    auto t = std::get<1>(x_in);

    if constexpr (Tag == ViewTag::LiftedCone)
      if (t < T(0)) return false;

    Point shift;
    if (r.has_value() && c.has_value()) {
      t *= r.value();
      shift = c.value();
    }

    // (dakre) negated norm function hypograph has opposite sign on the t
    // leaving the inequality unchanged
    if constexpr (Tag == ViewTag::HypographCone) t *= T(-1);

    const bool zeros = linalg::all_zero(x) && t == T(0);

    switch (Type) {
      case NormConeType::L2OC:
        return linalg::L2<Point>()(x - shift) < t;
      case NormConeType::L2CC:
        return zeros || linalg::L2<Point>()(x - shift) <= t;
      case NormConeType::L1OC:
        return linalg::L1<Point>()(x - shift) < t;
      case NormConeType::L1CC:
        return zeros || linalg::L1<Point>()(x - shift) <= t;
      case NormConeType::LinfOC:
        return linalg::Linf<Point>()(x - shift) < t;
      case NormConeType::LinfCC:
      default:
        return zeros || linalg::Linf<Point>()(x - shift) <= t;
    }
  }

  /**
   * @brief is_well_formed checks to see if norm cone is valid
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const { return true; }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return other.self().c == c && other.self().r == r;
  }

  /**
   * @brief transform impl
   *
   * @return self
   */
  template <linalg::MatrixLike Mat>
  constexpr auto transform_impl(const affine::AffineMap<Mat>& /* map */) const {
    return *this;
  }

  /**
   * @brief cone view impl
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    if constexpr (Tag == ViewTag::ConvexSet)
      return ConeView(NormCone<Space, Type, ViewTag::LiftedCone>());
    else
      return ConeView(NormCone<Space, Type, Tag>());
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
  projective_map_impl(const ProjectiveMap<MatA, MatC>& /*map*/,
                      const Point& /*x*/) const {
    return std::nullopt;
  }

 private:
  /** @brief member variables in the norm cone */
  std::optional<Point> c;  //< center point from norm ball -> cone
  std::optional<T> r;      //< radius from norm ball -> cone
};

}  // namespace convex
