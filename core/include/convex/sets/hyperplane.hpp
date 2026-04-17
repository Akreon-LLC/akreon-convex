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

namespace convex {

/// @brief alias for domain on hyperplane
template <linalg::RnSpace Space>
using HyperplaneDom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for codomain on halfspace
template <linalg::RnSpace Space>
using HyperplaneCodom = HyperplaneDom<Space>;

/// @brief alias for meta on hyperplane
template <linalg::RnSpace Space>
using HyperplaneMeta =
    ConvexAtomMeta<HyperplaneDom<Space>, HyperplaneCodom<Space>,
                   CurvatureType::CONVEX>;

/**
 * @brief Hyperplane defition defined on valid vector space
 *
 * H = { x | a^T x = b } \equiv { x | a^T x - b = 0 }
 */
template <typename Meta, typename Child>
struct Hyperplane : ConvexSet<Meta, Hyperplane<Meta, Child>, Child> {
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Meta, Hyperplane<Meta, Child>, Child>;
  using ConvexBase::N;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /**
   * @brief CTOR for hyperplane
   *
   * @param Point (i.e., n-dim vec of type T)
   * @param b bias scalar
   */
  constexpr explicit Hyperplane(const PointT& a_in, const T& b_in,
                                const Child& child_in)
      : ConvexBase(child_in), a(a_in), b(b_in) {}

  /**
   * @brief contains function to see if x lies in the hyerplane defined by
   * a^T x = b
   *
   * @param point x of type T (n-dim)
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const PointT& x, const T& tol) const {
    return linalg::constexpr_abs(linalg::dot(a, x) - b) < tol;
  }

  /**
   * @brief is well formed checks to see if the hyperplane is valid
   *
   * @return T/F
   */
  constexpr bool is_well_formed() const {
    if (linalg::all_zero(a)) return false;
    return true;
  }

  /**
   * @brief transform implementation
   *
   * H = { x | a^T x = b }
   * f(H) = { y = f(x) | x \in H }
   * y = U x + t -> x = U^{-1} (y - t)
   * a^T U^{-1} (y - t) = b \to a^T U^{-1} y = b + a^T U^{-1}
   *
   * @param affine map
   *
   * @return transformed hyperplane
   */
  template <linalg::SquareMatrixLike Mat>
    requires(linalg::mat_cols_v<Mat> == N &&
             std::same_as<linalg::mat_type_t<Mat>, T>)
  constexpr auto transform_impl(const affine::AffineMap<Mat>& map) const {
    const auto [U_inv, event] = linalg::MatrixInverse<Mat>()(map.linear_part());
    static_cast<void>(event);
    const auto aT_U = transpose(as_col_vector(a)) * U_inv;
    return Hyperplane<Meta, Child>(as_vector(aT_U),
                                   b + (aT_U * map.translation())(0),
                                   std::get<0>(this->children));
  }

  /**
   * @brief is_equal implementation
   *
   * @return T/F
   */
  template <typename Other>
  constexpr bool is_equal_impl(const Other& other) const {
    return other.self().a == a && other.self().b == b;
  }

  /**
   * @brief canonicalize impl
   *
   * Apply P^{-1}(C) -> K and map K to cone block
   *
   * a^T x = b
   *
   * @return cone block
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize_set(const Builder& builder) const {
    static_cast<void>(Type);
    const auto child_builder =
        std::get<0>(this->children)
            .template canonicalize<AtomType::SET>(builder);
    const auto child_coord = child_builder.coord_map;
    const auto child_cb = child_builder.cb;
    const auto child_eq = child_builder.eq_map;

    static constexpr size_t ChildCodom =
        linalg::mat_cols_v<decltype(child_coord.linear_part())>;
    static_assert(ChildCodom == N,
                  "Convex set domain does not equal the child coordinate "
                  "linear map's codomain in size");

    const auto eq_map = affine::AffineMap<linalg::RowVector<T, N>>(
                            linalg::as_row_vector(a), linalg::Vector<T, 1>{b})
                            .compose(child_coord);

    static constexpr size_t ChildEqRows =
        linalg::mat_rows_v<decltype(child_eq.linear_part())>;
    static constexpr size_t EqRows = ChildEqRows + 1;
    static constexpr size_t EqCols =
        linalg::mat_cols_v<decltype(eq_map.linear_part())>;

    linalg::Matrix<T, EqRows, EqCols> A{};
    linalg::Vector<T, EqRows> g{};

    if constexpr (ChildEqRows != 0) {
      A = linalg::set_block<0, ChildEqRows, 0, EqCols>(A,
                                                       child_eq.linear_part());
      for (size_t i = 0; i < ChildEqRows; ++i) g(i) = child_eq.translation()(i);
    }

    A = linalg::set_block<ChildEqRows, EqRows, 0, EqCols>(A,
                                                          eq_map.linear_part());
    g(ChildEqRows) = eq_map.translation()(0);

    const auto new_eq_map = affine::AffineMap<decltype(A)>(A, g);
    return conic::CanonicalFormBuilder<
        conic::builder_dom_t<decltype(builder)>, decltype(child_coord),
        decltype(child_cb), decltype(new_eq_map)>(child_coord, child_cb,
                                                  new_eq_map);
  }

  /** @brief member var in hyperplane */
  PointT a;  //< n-dim vec of type T
  T b;       //< bias scalar of type T
};

template <typename Point, typename T, typename Child>
Hyperplane(const Point&, const T&, const Child&)
    -> Hyperplane<
        HyperplaneMeta<dom_space_t<meta_codom_t<atom_meta_t<Child>>>>,
        atom_decay_t<Child>>;

}  // namespace convex
