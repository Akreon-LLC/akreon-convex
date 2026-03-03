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
#include "linalg/operator/symmetric_matrix.hpp"

namespace convex {

/// @brief alias for domain on ellipsoid
template <linalg::RnSpace Space>
using EllipsoidDom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for domain on ellipsoid
template <linalg::RnSpace Space>
using EllipsoidCodom = EllipsoidDom<Space>;

/// @brief alias for meta on ellipsoid
template <linalg::RnSpace Space>
using EllipsoidMeta = ConvexAtomMeta<EllipsoidDom<Space>, EllipsoidCodom<Space>,
                                     CurvatureType::CONVEX>;

/**
 * @brief Ellipsoid defined as the following
 *
 * E = { x | (x - c)^T P (x - c) <= 1 } s.t. P \in S^n_+
 */
template <typename Meta, typename Child>
struct Ellipsoid : ConvexSet<Meta, Ellipsoid<Meta, Child>, Child> {
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Meta, Ellipsoid<Meta, Child>, Child>;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::N;

  /// @brief common mat type for P
  using Mat = linalg::Matrix<T, N, N>;

  /**
   * @brief CTOR for ellipsoidal set
   *
   * @param center point (i.e., n-dim vector of type T)
   * @param P \in S^n_+
   * @param child
   */
  constexpr explicit Ellipsoid(const PointT& c_in, const Mat& P_in,
                               const Child& child_in)
      : ConvexBase(child_in), c(c_in), P(P_in) {}

  /**
   * @brief contains function to see if x is in the ellipsoid (x - c)^T P^{-1}
   * (x - c) <= 1
   *
   * @param point x of type T with n elements
   * @param tolerance of tyep T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const PointT& x, const T& /* tol */) const {
    const auto delta = x - c;
    const auto delta_T = linalg::transpose(as_col_vector(delta));
    const auto res = (delta_T * (linalg::inverse(P) * delta));
    static_assert(linalg::vec_size_v<decltype(res)> == 1);
    return res(0) <= T(1);
  }

  /**
   * @brief is_well_formed implementation checks to see if set is valid
   *
   * @return T/F
   */
  constexpr bool is_well_formed() const {
    linalg::SymmetricMatrix S(P);
    return S.type == linalg::SymmetricOperatorType::PD;
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
    return Ellipsoid<Meta, Child>(map.linear_part() * c + map.translation(),
                                  transpose(U_inv) * P * U_inv,
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
    return other.self().c == c && other.self().P == P;
  }

  /**
   * @brief canonicalize impl
   *
   * Apply P^{-1}(C) -> K and map K to cone block
   *
   * E = { x | (x - c)^T P (x - c) <= 1, P \in S^n_+ }
   *
   * [old] homogenzed t deprecated
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
   * ----
   * { (x, u, 1) | u^T(x) D u(x) <= 1 }
   * 1 - u^T(x) D u(x) \in rplus
   * so... K_soc = { (x,1) | ||D^{1/2} V^T (x - c) ||_2 <= 1 }
   * [ D^{1/2} V^T 0 ] [ x; 1 ] + [ -D^{1/2} V^T c ; 1 ] -> (x,t) in SOC
   *
   * @return cone block
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize_set(const Builder& builder) const {
    const auto child_builder =
        std::get<0>(this->children)
            .template canonicalize<AtomType::SET>(builder);
    const auto child_coord = child_builder.coord_map;
    const auto child_cb = child_builder.cb;

    static constexpr size_t ChildCodom =
        linalg::mat_cols_v<decltype(child_coord.linear_part())>;
    static_assert(ChildCodom == N,
                  "Convex set domain does not equal the child coordinate "
                  "linear map's codomain in size");

    linalg::Matrix<T, N + 1, N> F;
    linalg::Vector<T, N + 1> g;
    const auto eig_decomp = linalg::EigenDecomposition(P);
    const auto& D = eig_decomp.lambda;
    linalg::Matrix<T, N, N> D_sqrt;
    for (size_t i = 0; i < N; ++i)
      D_sqrt(i)(i) = linalg::constexpr_sqrt_safe(D(i).real());

    linalg::Matrix<T, N, N> VT;
    const auto& V = eig_decomp.V;
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j) VT(j)(i) = V(i)(j).real();

    const auto w = D_sqrt * VT;
    F = linalg::set_block<0, N, 0, N>(F, w);
    const auto wc = T(-1) * w * c;
    for (size_t i = 0; i < N; ++i) g(i) = wc(i);
    g(N) = T(1);
    const auto map = affine::AffineMap<decltype(F)>(F, g);
    const auto soc = conic::SecondOrderCone<linalg::Vector<T, N + 1>>();
    const auto cones = std::make_tuple(soc);

    const auto new_builder =
        conic::ConeBlockBuilder<conic::builder_dom_t<decltype(builder)>, decltype(map),
                         decltype(child_cb)>(map, child_cb);
    return new_builder.template make<Type>(cones);
  }

  /** @brief member variables in the ellipsoid */
  PointT c;  //< n-dim vector of type T representing center of ellipsoid
  Mat P;     //< square matrix representing directions for ellipsoid (valid \in
             // S^n_+)
};

}  // namespace convex
