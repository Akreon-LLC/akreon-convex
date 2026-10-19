/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "convex_set.hpp"
#include "akreon/linalg/operator/symmetric_matrix.hpp"

namespace convex {

/// @brief alias for domain on ellipsoid
template <linalg::RnSpace Space>
using EllipsoidDom = ConvexDomain<Space, VarType::FREE>;

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
 * E = { x | (x - c)^T P (x - c) <= 1 } s.t. P in S^n_+
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
   * @param P in S^n_+
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
  constexpr bool contains_impl(const PointT& x, T /* tol */) const {
    const auto delta = x - c;
    const auto delta_T = linalg::transpose(as_col_vector(delta));
    const auto [P_inv, event] = linalg::MatrixInverse<decltype(P)>()(P);
    static_cast<void>(event);
    const auto res = (delta_T * (P_inv * delta));
    static_assert(linalg::vector_dim_v<decltype(res)> == 1);
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
   * f(x) = U x + t -> { y = f(x) | x in E }
   * x = U^{-1}(y - t), sub back in we get...
   * E' = {y | (U^{-1}(y - t) - c)^T P (U^{-1}(y - t) - c) <= 1}
   * ... U^{-1}(y - t) - c to U^{-1}(y - (U c + t)
   * E' = { y | (y - (U c + t))^T U^{-T} P U^{-1} (y - (U c + t)) <= 1
   *
   * c' = U c + t
   * P' = U^{-T} P u
   *
   * @param Affine map
   *
   * @return transformed ellipsoid
   */
  template <linalg::SquareMatrix Mat>
    requires(linalg::mat_cols_v<Mat> == N &&
             std::same_as<T, linalg::matrix_scalar_t<Mat>>)
  constexpr auto transform_impl(const affine::AffineMap<Mat>& map) const {
    const auto [U_inv, event] = linalg::MatrixInverse<Mat>()(map.linear_part());
    static_cast<void>(event);
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
  template <typename Other>
  constexpr bool is_equal_impl(const Other& other) const {
    return other.self().c == c && other.self().P == P;
  }

  /**
   * @brief canonicalize impl
   *
   * E = { x | (x - c)^T P (x - c) <= 1, P in S^n_+ }
   *
   * Eigen decomposition: P = V D V^T
   * (x - c)^T V D V^T (x - c) <= 1
   * let u = V^T (x - c), u^T = (x-c)^T V
   *
   * Let w = D^{1/2} V^T (x - c). Then
   *   (x - c)^T P (x - c) <= 1  <=>  ||w||_2 <= 1.
   *
   * Direct cone map:
   *   [ D^{1/2} V^T ] x + [ -D^{1/2} V^T c ]
   *   [      0      ]     [          1     ]
   *   in K_soc
   *
   * @return cone block
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize_set(const Builder& builder) const {
    const auto child_builder =
        std::get<0>(this->children)
            .template canonicalize<AtomType::SET>(builder);
    const auto child_coord = child_builder.coord_map;

    static constexpr std::size_t ChildCodom =
        linalg::mat_cols_v<decltype(child_coord.linear_part())>;
    static_assert(ChildCodom == N,
                  "Convex set domain does not equal the child coordinate "
                  "linear map's codomain in size");

    linalg::Matrix<T, N + 1, N> F;
    linalg::Vector<T, N + 1> g;
    const auto eig_decomp = linalg::EigenDecomposition(P);
    const auto& D = eig_decomp.lambda;
    linalg::Matrix<T, N, N> D_sqrt;
    for (std::size_t i = 0; i < N; ++i)
      D_sqrt(i, i) = linalg::constexpr_sqrt_safe(D(i).real());

    linalg::Matrix<T, N, N> VT;
    const auto& V = eig_decomp.V;
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = 0; j < N; ++j) VT(j, i) = V(i, j).real();

    const auto w = D_sqrt * VT;
    F = linalg::set_block<0, N, 0, N>(F, w);
    const auto wc = T(-1) * w * c;
    for (std::size_t i = 0; i < N; ++i) g(i) = wc(i);
    g(N) = T(1);
    const auto map = affine::AffineMap<decltype(F)>(F, g);
    const auto soc = conic::SecondOrderCone<linalg::Vector<T, N + 1>>();
    const auto cones = std::make_tuple(soc);
    const auto emission =
        conic::LocalEmission(map, cones, child_builder.eq_map);

    return child_builder.template realize<Type>(emission);
  }

  /** @brief member variables in the ellipsoid */
  PointT c; //< n-dim vector of type T representing center of ellipsoid
  Mat P;    //< square matrix representing directions for ellipsoid (valid in
            // S^n_+)
};

template <typename Point, typename Mat, typename Child>
Ellipsoid(const Point&, const Mat&, const Child&)
    -> Ellipsoid<EllipsoidMeta<dom_space_t<meta_codom_t<atom_meta_t<Child>>>>,
                 atom_decay_t<Child>>;

} // namespace convex
