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

 private:
  /** @brief member variables in the ellipsoid */
  Point c;  //< n-dim vector of type T representing center of ellipsoid
  linalg::Matrix<T, N, N> P;  //< square matrix representing directions for
                              // ellipsoid (valid \in S^n_+)
  EllipsoidType type;         //< ellipsoid type
};

}  // namespace convex
