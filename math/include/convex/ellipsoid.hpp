/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set.hpp"

namespace convex {

/// @brief simple enum to describe type of ellipsoid
enum struct EllipsoidType { CE, OE };

/**
 * @brief Ellipsoid defined as the following
 *
 * E = { x | (x - c)^T P (x - c) <= 1 } s.t. P \in S^n_+ (symmetric positive
 * definite)
 */
template <linalg::VectorSpace Space>
  requires(linalg::space_dim_v<Space> ==
               linalg::vec_size_v<linalg::space_point_t<Space>> &&
           !linalg::is_complex_v<linalg::space_point_t<Space>>)
class Ellipsoid : public ConvexSet<Space, Ellipsoid<Space>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, Ellipsoid<Space>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::ScalarT;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief CTOR for ellipsoidal set
   *
   * @param Point (i.e., n-dim vector of type T)
   * @param P \in S^n_+
   * @param enum type describing ellipsoid
   */
  constexpr explicit Ellipsoid(const Point& c_in,
                               const linalg::Matrix<T, N, N>& P_in,
                               const EllipsoidType& type_in)
      : c(c_in), P(P_in), type(type_in) {}

 private:
  /// @brief allow base class to invoke impl pure virtual (v-table like) funcs
  friend ConvexBase;

  /**
   * @brief contains function to see if x is in the ellipsoid (x - c)^T P (x -
   * c) <= 1

   * @param point x of type T with n elements
   * @param tolerance of tyep T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& tol) const {
    (void)tol;
    const auto delta = as_col_vector(x - c);
    const auto delta_t = linalg::transpose(delta);

    const auto val = (delta_t * (linalg::inverse(P) * delta));
    static_assert(linalg::mat_rows_v<decltype(val)> == 1 &&
                  linalg::mat_cols_v<decltype(val)> == 1);
    switch (type) {
      case EllipsoidType::OE:
        return val(0)(0) < ScalarT(1);
      case EllipsoidType::CE:
      default:
        return val(0)(0) <= ScalarT(1);
    }
    return true;
  }

  /**
   * @brief is_well_formed implementation checks to see if set is valid
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const {
    linalg::SymmetricMatrix S(P);
    return S.type == linalg::SymmetricMatrixType::PD ||
           S.type == linalg::SymmetricMatrixType::PSD;
  }

  /**
   * @brief transform set from an affine map
   *
   * E = {x | (x - c)^T P (x - c) <= 1}
   * f(x) = U x + t -> { y = f(x) | x \in E }
   * x = U^{-1}(y - t), sub back in we get...
   * E' = {y | (U^{-1}(y - t) - c)^T P (U^{-1}(y - t) - c) <= 1}
   * ... U^{-1}(y - t) - c \to U^{-1}(y - (U c + t)
   * E' = { y | (y - (U c + t))^T U^{-T} P U^{-1} (y - (U c + t)) <= 1
   *
   * c' = U c + t
   * P' = U^{-T} P u
   *
   * @param Affine map
   *
   * @return transformed ellipsoid
   */
  template <linalg::SquareMatrixLike Mat>
    requires(linalg::mat_rows_v<Mat> == N)
  constexpr auto transform_impl(const AffineMap<Mat>& map) const {
    const auto U_inv = linalg::inverse(map.linear_part());
    return Ellipsoid<Space>(map.linear_part() * c + map.translation(),
                            transpose(U_inv) * P * U_inv, type);
  }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(
      const ConvexSet<Space, Ellipsoid<Space>>& other) const {
    return other.self().c == c && other.self().P == P &&
           other.self().type == type;
  }

  /** @brief member variables in the ellipsoid */
  Point c;  //< n-dim vector of type T representing center of ellipsoid
  linalg::Matrix<T, N, N> P;  //< square matrix representing directions for
                              // ellipsoid (valid \in S^n_+)
  EllipsoidType type;         //< ellipsoid type
};

}  // namespace convex
