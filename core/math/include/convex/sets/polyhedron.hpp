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

#include "halfspace.hpp"
#include "hyperplane.hpp"

namespace convex {

/// @brief alias for domain on poly
template <linalg::RnSpace Space>
using PolyhedronDom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for codomain on poly
template <linalg::RnSpace Space>
using PolyhedronCodom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for meta on poly
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using PolyhedronMeta =
    ConvexAtomMeta<PolyhedronDom<Dom>, PolyhedronCodom<Codom>,
                   CurvatureType::CONVEX>;

/**
 * @brief Polyhedron convex set
 *
 * P = { x | A x <= b }... deprecating Cx = d in favor of a separate
 * intersection with an affine subspace or additional constraint to be added in
 * the problem space
 */
template <typename Meta, typename Child, HalfspaceType Type>
struct Polyhedron : ConvexSet<Meta, Polyhedron<Meta, Child, Type>, Child> {
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Meta, Polyhedron<Meta, Child, Type>, Child>;
  using ConvexBase::M;
  using ConvexBase::N;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /// @brief bring into scope type traits from MatA
  using Mat = linalg::Matrix<T, M, N>;
  using Bias = linalg::Vector<T, M>;
  using Space = linalg::Matrix<T, 1, N>;

  /**
   * @brief CTOR for polyhedron set
   *
   * @param A basis of vectors describing linear combinations with x to form
   * halfspace
   * @param b vector describing halfspace inequality (per row vec in A)
   * @param child
   */
  constexpr explicit Polyhedron(const Mat& A_in, const Bias& b_in,
                                const Child& child_in)
      : ConvexBase(child_in), A(A_in), b(b_in) {}

  /**
   * @brief is_well_formed implementation
   *
   * @return T/F
   */
  constexpr bool is_well_formed() const {
    for (size_t i = 0; i < M; ++i)
      if (!Halfspace<HalfspaceMeta<Space>, Child, Type>(
               A(i), b(i), std::get<0>(this->children))
               .is_well_formed())
        return false;
    return true;
  }

  /**
   * @brief check if candidate vector is in the set
   *
   * @param Vec X of type T with N elements
   *
   * @return T/F
   */
  constexpr bool contains_impl(const PointT& x, const T& tol) const {
    for (size_t i = 0; i < M; ++i) {
      const auto half = Halfspace<HalfspaceMeta<Space>, Child, Type>(
          A(i), b(i), std::get<0>(this->children));
      if (!half.contains(x, tol)) return false;
    }
    return true;
  }

  /**
   * @brief transform polyhedron by affine map
   *
   * P = { x | Ax <= b, Cx = d }
   * f(x) = U x + t
   *
   * P' = { y = f(x) | x \in P }
   * y = U x + t \to x = U^{-1}(y - t)
   *
   * constraints become: A U^{-1} (y - t) <= b & C U^{-1} (y - t) = d
   * (i) A U^{-1} y <= b + A U^{-1} t
   * (ii) C U^{-1} y = d + C U^{-1} t
   *
   * @return T/F
   */
  template <linalg::SquareMatrixLike Mat>
    requires(std::same_as<T, linalg::mat_type_t<Mat>> &&
             linalg::mat_cols_v<Mat> == N)
  constexpr auto transform_impl(const affine::AffineMap<Mat>& map) const {
    const auto& U = map.linear_part();
    const auto& trans = map.translation();

    const auto U_inv = linalg::inverse(U);
    const auto A_new = A * U_inv;
    const auto b_new = b + A_new * trans;

    return Polyhedron<Meta, Child, Type>(A_new, b_new,
                                         std::get<0>(this->children));
  }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return other.self().A == A && other.self().b == b;
  }

  /**
   * @brief canonicalize impl
   *
   * Apply P^{-1}(C) -> K and map K to cone block
   *
   * C = { x | A x <= b, Cx = d }
   * K = { (x, t) | A x <= b t, Cx = d t, t >= 0 }
   *
   * let z = C x - d t = 0... z in zoc
   * let w = A x - b t <= 0... w in rneg
   * t in rplus
   *
   * [ A -b ] [ x; t ] <= 0
   * [ 0  1 ]
   *
   * [ C -d ] [ x; t ] = 0
   * [ 0  1 ]
   *
   * @return cone block
   */
  constexpr auto canonicalize_impl() const {
    std::tuple<> empty;
    static constexpr size_t Codom = 2 * M;
    static constexpr size_t Dom = N + 1;
    linalg::Matrix<T, Codom, Dom> F;
    linalg::Vector<T, Codom> g;
    conic::ConeBlock cb(affine::AffineMap(F, g), empty);
    return canonicalize_halfspaces<0>(cb);
  }

  /**
   * @brief canonicalize halfspace
   *
   * @param cone_block
   *
   * @return halfspaces
   */
  template <size_t i, affine::AffineMapLike Map, conic::ConeLike... Cs>
  constexpr auto canonicalize_halfspaces(
      const conic::ConeBlock<Map, Cs...>& cone_block) const {
    if constexpr (i >= M)
      return cone_block;
    else {
      const auto halfspace =
          Halfspace<HalfspaceMeta<Space>, Child, Type>(A(i), b(i));
      const auto cb = halfspace.canonicalize();
      auto F = cone_block.map.linear_part();
      const auto g = cone_block.map.translation();
      static constexpr size_t j = 2 * i;
      F = linalg::set_block<j, j + 2, 0, N + 1>(F, cb.map.linear_part());
      const auto map = affine::AffineMap<decltype(F)>(F, g);
      const auto cones = std::tuple_cat(cone_block.cones, cb.cones);
      return canonicalize_halfspaces<i + 1>(conic::ConeBlock(map, cones));
    }
  }

  /** @brief member variables in the polyhedron */
  Mat A;   //< set of finite number of halfspaces
  Bias b;  //< codomain vector describing halfspace inequalities
};
}  // namespace convex
