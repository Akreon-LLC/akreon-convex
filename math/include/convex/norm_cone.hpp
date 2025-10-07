/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "norm_ball.hpp"

namespace convex {

enum struct NormConeType { L2OC, L2CC, L1OC, L1CC, LinfOC, LinfCC };

/**
 * @brief Norm cone definition valid on a vector space dim(R^n + 1)
 *
 * { (x, t) | ||x|| <= t }... closed norm cone defined on L1, L2, Linf
 * { (x, t) | ||x|| < t }... open cone defined on L1, L2, Linf
 */
template <linalg::VectorSpace Space, NormConeType Type>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class NormCone : public ConvexSet<Space, NormCone<Space, Type>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, NormCone<Space, Type>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief alias for scalar type of underlying set type
  using ScalarT = linalg::scalar_type_t<T>;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief CTOR for norm cone
   *
   * @param t (scalar rep variable radius)
   */
  constexpr explicit NormCone(const ScalarT& t_in) : t(t_in) {}

 protected:
  /// @brief allow base class to invoke impls
  friend ConvexBase;

  /**
   * @brief contains function to see if x is in the norm cone (y, t) \in K
   *
   * K = {(x, t) \in R^n x R | ||x|| <= t}
   *
   * @param point x
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& /* tol */) const {
    using T = linalg::vec_type_t<Point>;
    switch (Type) {
      case NormConeType::L2OC:
        return linalg::L2<T, N>()(x) < t;
      case NormConeType::L2CC:
        return linalg::L2<T, N>()(x) <= t;
      case NormConeType::L1OC:
        return linalg::L1<T, N>()(x) < t;
      case NormConeType::L1CC:
        return linalg::L1<T, N>()(x) <= t;
      case NormConeType::LinfOC:
        return linalg::Linf<T, N>()(x) < t;
      case NormConeType::LinfCC:
      default:
        return linalg::Linf<T, N>()(x) <= t;
    }
  }

  /**
   * @brief is_well_formed checks to see if norm cone is valid
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const { return t >= 0; }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return other.self().t == t;
  }

  /**
   * @brief transform impl
   *
   * Affine map of a cone will break cone property so no-op here for the time
   * being. Cones are homogenous sets anchored at the origin and don't "move"
   * with affine maps like other convex sets
   *
   * @param Affine map
   *
   * @return transformed convex set
   */
  template <linalg::SquareMatrixLike Mat>
    requires(linalg::mat_cols_v<Mat> == N &&
             std::same_as<T, linalg::mat_type_t<Mat>>)
  constexpr auto transform_impl(const AffineMap<Mat>& /*map*/) const {
    return *this;
  }

  /** @brief member variables in the norm cone */
  ScalarT t;  //< scalar of type T representing threshold for cone
};

}  // namespace convex
