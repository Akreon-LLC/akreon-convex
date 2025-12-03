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
#include "linalg/symmetric_matrix.hpp"

namespace convex {

/**
 * @brief Ellipsoid defined as the following
 *
 * E = { x | (x - c)^T P (x - c) <= 1 } s.t. P \in S^n_+
 */
template <linalg::RnSpace Space, EllipsoidType Type>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Ellipsoid : public ConvexSet<Space, Ellipsoid<Space, Type>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, Ellipsoid<Space, Type>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;
  using ConvexBase::N;

  /// @brief common mat type for P
  using Mat = linalg::Matrix<T, N, N>;

  /**
   * @brief CTOR for ellipsoidal set
   *
   * @param center point (i.e., n-dim vector of type T)
   * @param P \in S^n_+
   */
  constexpr explicit Ellipsoid(const Point& c_in, const Mat& P_in)
      : c(c_in), P(P_in) {}

  /**
   * @brief contains function to see if x is in the ellipsoid (x - c)^T P^{-1}
   * (x - c) <= 1
   *
   * @param point x of type T with n elements
   * @param tolerance of tyep T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& /* tol */) const {
    const auto delta = x - c;
    const auto delta_T = linalg::transpose(as_col_vector(delta));
    const auto res = (delta_T * (linalg::inverse(P) * delta));
    static_assert(linalg::vec_size_v<decltype(res)> == 1);

    switch (Type) {
      case EllipsoidType::OE:
        return res(0) < 1;
      case EllipsoidType::CE:
      default:
        return res(0) <= 1;
    }
    return false;
  }

  /**
   * @brief is_well_formed implementation checks to see if set is valid
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const {
    linalg::SymmetricMatrix S(P);
    return S.type == linalg::SymmetricMatrixType::PD;
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
    requires(linalg::mat_cols_v<Mat> == N &&
             std::same_as<T, linalg::mat_type_t<Mat>>)
  constexpr auto transform_impl(const affine::AffineMap<Mat>& map) const {
    const auto U_inv = linalg::inverse(map.linear_part());
    return Ellipsoid<Space, Type>(map.linear_part() * c + map.translation(),
                                  transpose(U_inv) * P * U_inv);
  }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return other.self().c == c && other.self().P == P;
  }

  /**
   * @brief canonicalize impl
   *
   * Apply P^{-1}(C) -> K and map K to cone block
   *
   * E = { x | (x - c)^T P (x - c) <= 1, P \in S^n_+ }
   *
   * Factor P = L L^T
   * let y = L^{-1}(x - c) in dom of RSOC
   * let x = Ly + c in orig coords
   *
   * SOC
   * K = { (y, t) | ||y||_2^2 <= t }
   *
   * [ L^{-1} 0 ] [ x; t] + [ -L^{-1} c ; 0 ] -> (L^{-1} (x - c), t)
   * [ 0      1 ]
   *
   * @return cone block
   */
#if 0
  constexpr auto canonicalize_impl() const {
    linalg::Matrix<T, M + 1, N + 1> F;
    F = linalg::set_block<0, M, 0, N>(F, A);
    F = linalg::set_block<0, M, N, N + 1>(F, linalg::as_col_vector(T(-1) * b));
    F[M][N] = T(1);
    linalg::Vector<T, M + 1> g;
    const auto map = affine::AffineMap<decltype(F)>(F, g);
    const auto zoc = conic::ZeroOrderCone<linalg::Matrix<T, M, N + 1>>();
    const auto rplus = conic::NonnegOrthantCone<linalg::Matrix<T, 1, N + 1>>();
    return conic::ConeBlock(map, zoc, rplus);
  }
#endif

 private:
  /** @brief member variables in the ellipsoid */
  Point c;  //< n-dim vector of type T representing center of ellipsoid
  Mat P;    //< square matrix representing directions for ellipsoid (valid \in
            // S^n_+)
};

}  // namespace convex
