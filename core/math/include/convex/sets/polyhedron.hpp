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

/**
 * @brief Polyhedron convex set
 *
 * P = { x | A x <= b, C x = d }
 */
template <linalg::RnSpace Space, HalfspaceType Type, linalg::MatrixLike MatA,
          linalg::MatrixLike MatC = linalg::Matrix<linalg::mat_type_t<Space>, 0,
                                                   linalg::space_dim_v<Space>>>
  requires(!linalg::is_complex_v<linalg::mat_type_t<Space>> &&
           !linalg::is_complex_v<linalg::mat_type_t<MatA>> &&
           !linalg::is_complex_v<linalg::mat_type_t<MatC>>)
class Polyhedron
    : public ConvexSet<Space, Polyhedron<Space, Type, MatA, MatC>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, Polyhedron<Space, Type, MatA, MatC>>;
  using ConvexBase::Dim;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief helper to get N elements in point type
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /// @brief bring into scope type traits from MatA
  using MatA_T = linalg::mat_type_t<MatA>;
  static constexpr size_t RowsA = linalg::mat_rows_v<MatA>;
  static constexpr size_t ColsA = linalg::mat_cols_v<MatA>;

  /// @brief bring into scope typetraits from MatC
  static constexpr size_t RowsC = linalg::mat_rows_v<MatC>;
  static constexpr size_t ColsC = linalg::mat_cols_v<MatC>;

  static_assert(ColsA == ColsC, "domain mismatch between cols dim of A and C");

  /// @brief bring into scope type traits from MatC
  using MatC_T = linalg::mat_type_t<MatC>;

  /// @brief verify types align
  static_assert(std::same_as<T, MatA_T>);
  static_assert(std::same_as<MatA_T, MatC_T>);

  /**
   * @brief CTOR for polyhedron set
   *
   * @param A basis of vectors describing linear combinations with x to form
   * halfspace
   * @param b vector describing halfspace inequality (per row vec in A)
   * @param C basis of vectors describing linear combinations with x to form
   * hyperplane
   * @param d vector describing hyperplane bias equality (per row vec in C)
   */
  constexpr explicit Polyhedron(
      const MatA& A_in, const linalg::Vector<T, RowsA>& b_in,
      MatC C_in = MatC{},
      linalg::Vector<T, RowsC> d_in = linalg::Vector<T, RowsC>{})
      : A(A_in), b(b_in), C(C_in), d(d_in) {}

  /**
   * @brief is_well_formed implementation
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const {
    // (1) verify halfspaces for each row in A and b check if halfspace is
    // valid
    for (size_t i = 0; i < RowsA; ++i)
      if (!Halfspace<Space, Type>(A(i), b(i)).is_well_formed()) return false;

    // (2) verify hyperplanes C x = d is valid
    for (size_t i = 0; i < RowsC; ++i)
      if (RowsC > 0 && !Hyperplane<Space>(C(i), d(i)).is_well_formed())
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
  constexpr bool contains_impl(const Point& x, const T& tol) const {
    // (0) check if x is in the halfspace defined by A x - b t <= 0
    for (size_t i = 0; i < RowsA; ++i) {
      const auto cone = Halfspace<Space, Type>(A(i), b(i));
      if (!cone.contains(x, tol)) return false;
    }

    // (1) check if x is in the affine set defined by C x = d t
    for (size_t i = 0; i < RowsC; ++i) {
      const auto cone = Hyperplane<Space>(C(i), d(i));
      if (RowsC > 0 && !cone.contains(x, tol)) return false;
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
    const auto C_new = C * U_inv;
    const auto d_new = d + C_new * trans;

    return Polyhedron<Space, Type, decltype(A), decltype(C)>(A_new, b_new,
                                                             C_new, d_new);
  }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return other.self().A == A && other.self().b == b && other.self().C == C &&
           other.self().d == d;
  }

  /**
   * @brief get A mat
   *
   * @return A matrix
   */
  constexpr auto get_A() const { return A; }

  /**
   * @brief get C mat
   *
   * @return C matrix
   */
  constexpr auto get_C() const { return C; }

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
    static constexpr size_t Codom = 2 * (RowsA + RowsC);
    static constexpr size_t Dom = ColsA + 1;
    linalg::Matrix<T, Codom, Dom> F;
    linalg::Vector<T, Codom> g;
    conic::ConeBlock cb(affine::AffineMap(F, g), empty);
    const auto cb_half = canonicalize_halfspaces<0>(cb);
    return canonicalize_hyperplanes<0>(cb_half);
  }

 private:
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
    if constexpr (i >= RowsA)
      return cone_block;
    else {
      const auto halfspace = Halfspace<Space, Type>(A(i), b(i));
      const auto cb = halfspace.canonicalize();
      auto F = cone_block.map.linear_part();
      const auto g = cone_block.map.translation();
      static constexpr size_t j = 2 * i;
      F = linalg::set_block<j, j + 2, 0, ColsA + 1>(F, cb.map.linear_part());
      const auto map = affine::AffineMap<decltype(F)>(F, g);
      const auto cones = std::tuple_cat(cone_block.cones, cb.cones);
      return canonicalize_halfspaces<i + 1>(conic::ConeBlock(map, cones));
    }
  }

  /**
   * @brief canonicalize hyperplane
   *
   * @param cone_block
   *
   * @return hyperplanes
   */
  template <size_t i, affine::AffineMapLike Map, conic::ConeLike... Cs>
  constexpr auto canonicalize_hyperplanes(
      const conic::ConeBlock<Map, Cs...>& cone_block) const {
    if constexpr (i >= RowsC)
      return cone_block;
    else {
      const auto hyperplane = Hyperplane<Space>(C(i), d(i));
      const auto cb = hyperplane.canonicalize();
      auto F = cone_block.map.linear_part();
      const auto g = cone_block.map.translation();
      static constexpr size_t j = 2 * i + 2 * RowsA;
      F = linalg::set_block<j, j + 2, 0, ColsA + 1>(F, cb.map.linear_part());
      const auto map = affine::AffineMap<decltype(F)>(F, g);
      const auto cones = std::tuple_cat(cone_block.cones, cb.cones);
      return canonicalize_hyperplanes<i + 1>(conic::ConeBlock(map, cones));
    }
  }

  /** @brief member variables in the polyhedron */
  MatA A;  //< set of finite number of halfspaces
  linalg::Vector<T, RowsA>
      b;   //< codomain vector describing halfspace inequalities
  MatC C;  //< set of finite number of hyperplanes
  linalg::Vector<T, RowsC>
      d;  //< codomain vector describing bias for hyperplane equalities
};
}  // namespace convex
