/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
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

template <linalg::VectorSpace Space, HalfspaceType Type,
          linalg::MatrixLike MatA,
          linalg::MatrixLike MatC = linalg::Matrix<linalg::mat_type_t<Space>, 0,
                                                   linalg::space_dim_v<Space>>,
          ViewTag Tag = ViewTag::ConvexSet>
  requires(!linalg::is_complex_v<linalg::mat_type_t<Space>> &&
           !linalg::is_complex_v<linalg::mat_type_t<MatA>> &&
           !linalg::is_complex_v<linalg::mat_type_t<MatC>>)
class Polyhedron
    : public ConvexSet<Space, Polyhedron<Space, Type, MatA, MatC, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, Polyhedron<Space, Type, MatA, MatC, Tag>>;
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

  /// @brief bring into scope type traits from MatC
  using MatC_T = linalg::mat_type_t<MatC>;

  /// @brief verify types align
  static_assert(std::same_as<T, MatA_T>);
  static_assert(std::same_as<MatA_T, MatC_T>);

  /// @brief lifted point (based on lifted space or not)
  using AugmentedPoint = std::conditional_t<Tag == ViewTag::ConvexCone,
                                            linalg::Vector<T, N - 1>, Point>;

  /// @brief verify domain size align to vector space dim
  static constexpr size_t AugSize = linalg::vec_size_v<AugmentedPoint>;
  static_assert(AugSize == ColsA);
  static_assert(AugSize == ColsC);

  /// @brief mat space based on augmented dim size
  using MatSpace = linalg::Matrix<T, 1, AugSize>;

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
      if (!Halfspace<MatSpace, Type>(A(i), b(i)).is_well_formed()) return false;

    // (2) verify hyperplanes C x = d is valid
    for (size_t i = 0; i < RowsC; ++i)
      if (RowsC > 0 && !Hyperplane<MatSpace>(C(i), d(i)).is_well_formed())
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
  constexpr bool contains_impl(const Point& x_in, const T& tol) const {
    T t(1);
    AugmentedPoint x;
    if constexpr (Tag == ViewTag::ConvexCone) {
      if (x_in(N - 1) < T(0)) return false;
      t *= x_in(N - 1);
      x = std::get<0>(linalg::splice<N - 1>(x_in));
    } else {
      x = x_in;
    }

    // (0) check if x is in the halfspaces defined by A x <= b
    for (size_t i = 0; i < RowsA; ++i)
      if (!Halfspace<MatSpace, Type>(A(i), t * b(i)).contains(x, tol))
        return false;

    // (1) check if x is in the affine set defined by C x = d
    for (size_t i = 0; i < RowsC; ++i)
      if (RowsC > 0 && !Hyperplane<MatSpace>(C(i), t * d(i)).contains(x, tol))
        return false;

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
             linalg::mat_cols_v<Mat> == N && Tag == ViewTag::ConvexSet)
  constexpr auto transform_impl(const AffineMap<Mat>& map) const {
    const auto& U = map.linear_part();
    const auto& trans = map.translation();

    const auto U_inv = linalg::inverse(U);
    const auto A_new = A * U_inv;
    const auto b_new = b + A_new * trans;
    const auto C_new = C * U_inv;
    const auto d_new = d + C_new * trans;

    return Polyhedron<Space, Type, decltype(A), decltype(C), Tag>(A_new, b_new,
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
   * @brief inverse perspective func from convex set to cone
   *
   * @param t cone slice
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const
    requires(Tag == ViewTag::ConvexSet)
  {
    return ConeView(Polyhedron<linalg::Matrix<T, 1, N + 1>, Type, decltype(A),
                               decltype(C), ViewTag::ConvexCone>(A, b, C, d));
  }

  /**
   * @brief project impl
   *
   * @param project map
   * @param point in set
   *
   * @return projected set
   */
  template <linalg::MatrixLike MA, linalg::MatrixLike MC>
    requires(Tag == ViewTag::ConvexSet)
  constexpr std::optional<linalg::Vector<T, linalg::mat_cols_v<MA> + 1>>
  project_impl(const ProjectiveMap<MA, MC>& map, const Point& x) const {
    if (!this->contains(x)) return std::nullopt;
    return map.apply(x);
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

 private:
  /** @brief member variables in the polyhedron */
  MatA A;  //< set of finite number of halfspaces
  linalg::Vector<T, RowsA>
      b;   //< codomain vector describing halfspace inequalities
  MatC C;  //< set of finite number of hyperplanes
  linalg::Vector<T, RowsC>
      d;  //< codomain vector describing bias for hyperplane equalities
};
}  // namespace convex
