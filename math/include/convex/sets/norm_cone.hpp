/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
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
          ViewTag Tag = ViewTag::ConvexSet>
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

  /// @brief lifted point (based on lifted space or not)
  static constexpr size_t AugSize = N - 1;
  using AugmentedPoint = linalg::Vector<T, AugSize>;

  /**
   * @brief CTOR for norm cone
   */
  constexpr explicit NormCone() {}

  /**
   * @brief CTOR for norm cone (via norm ball conic view)
   *
   * @param center point from norm ball
   * @param r (radius from norm ball)
   */
  constexpr explicit NormCone(const AugmentedPoint& c_in, const T& r_in)
      : c(c_in), r(r_in) {}

  /**
   * @brief contains function to see if x is in the norm cone (y, t) \in K
   *
   * K = {(x, t) \in R^n x R | ||x|| <= t}
   *
   * @param point (x,t) - lifted by default
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x_in, const T& /* tol */) const {
    T t = x_in(N - 1);
    if (t < T(0)) return false;
    AugmentedPoint x = std::get<0>(linalg::splice<N - 1>(x_in));

    AugmentedPoint shift;
    if (r.has_value() && c.has_value()) {
      t *= r.value();
      shift = c.value();
    }

    const bool zeros = linalg::all_zero(x_in);

    switch (Type) {
      case NormConeType::L2OC:
        return linalg::L2<T, AugSize>()(x - shift) < t;
      case NormConeType::L2CC:
        return zeros || linalg::L2<T, AugSize>()(x - shift) <= t;
      case NormConeType::L1OC:
        return linalg::L1<T, AugSize>()(x - shift) < t;
      case NormConeType::L1CC:
        return zeros || linalg::L1<T, AugSize>()(x - shift) <= t;
      case NormConeType::LinfOC:
        return linalg::Linf<T, AugSize>()(x - shift) < t;
      case NormConeType::LinfCC:
      default:
        return zeros || linalg::Linf<T, AugSize>()(x - shift) <= t;
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
    requires(Tag == ViewTag::ConvexSet)
  constexpr auto transform_impl(const AffineMap<Mat>& /* map */) const {
    return *this;
  }

  /**
   * @brief cone view impl
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const
    requires(Tag == ViewTag::ConvexSet)
  {
    return ConeView(NormCone<Space, Type, ViewTag::ConvexCone>());
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
    requires(Tag == ViewTag::ConvexSet)
  constexpr std::optional<linalg::Vector<T, linalg::mat_cols_v<MatA> + 1>>
  project_impl(const ProjectiveMap<MatA, MatC>& /*map*/,
               const Point& /*x*/) const {
    return std::nullopt;
  }

 private:
  /** @brief member variables in the norm cone */
  std::optional<AugmentedPoint> c;  //< center point from norm ball -> cone
  std::optional<T> r;               //< radius from norm ball -> cone
};

}  // namespace convex
