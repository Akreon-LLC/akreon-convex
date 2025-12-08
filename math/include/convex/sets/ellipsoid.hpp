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
   * Eigen decomposition: P = V D V^T
   * (x - c)^T V D V^T (x - c) <= 1
   * let u = V^T (x - c), u^T = (x-c)^T V
   *
   * P^{-1}(E) = { (x, t) | (x - c)^T V D V^T (x - c) <= t^2 }
   * { (x, u, t) | u^T(x) D u(x) <= t^2 }
   *
   * Transform to SOC cone
   * let w = D^{1/2} u, D^{1/2} = diag(sqrt(lam_1), ... sqrt(lam_n))
   * u^T D u = (D^{1/2} u)^T (D^{1/2} u) = ||w||_2^2
   * K_soc = { (x,t) | ||D^{1/2} V^T (x - c)||_2 <= t }
   *
   * [ D^{1/2} V^T 0 ] [ x; t ] + [ -D^{1/2} V^T c ; 0 ]
   * [ 0           1 ]
   *
   * @return cone block
   */
  constexpr auto canonicalize_impl() const {
    linalg::Matrix<T, N + 1, N + 1> F;
    linalg::Vector<T, N + 1> g;
    const auto eig_decomp = linalg::EigenDecomposition(P);
    const auto D = eig_decomp.get_eigenvalues();
    linalg::Matrix<T, N, N> D_sqrt;
    for (size_t i = 0; i < N; ++i)
      D_sqrt[i][i] = linalg::constexpr_sqrt(D(i).real());

    linalg::Matrix<T, N, N> VT;
    const auto V = eig_decomp.get_eigenvectors();
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j) VT[j][i] = V(i)(j).real();

    const auto w = D_sqrt * VT;
    F = linalg::set_block<0, N, 0, N>(F, w);
    F[N][N] = T(1);
    const auto wc = T(-1) * w * c;
    for (size_t i = 0; i < N; ++i) g[i] = wc(i);
    const auto map = affine::AffineMap<decltype(F)>(F, g);
    const auto soc = conic::SecondOrderCone<decltype(F)>();
    return std::tuple(conic::ConeBlock(map, soc));
  }

 private:
  /** @brief member variables in the ellipsoid */
  Point c;  //< n-dim vector of type T representing center of ellipsoid
  Mat P;    //< square matrix representing directions for ellipsoid (valid \in
            // S^n_+)
};

}  // namespace convex
