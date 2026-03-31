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
using PolyhedronDom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for codomain on poly
template <linalg::RnSpace Space>
using PolyhedronCodom = ConvexDomain<Space, VarType::FREE>;

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
template <typename Meta, typename Child>
struct Polyhedron : ConvexSet<Meta, Polyhedron<Meta, Child>, Child> {
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Meta, Polyhedron<Meta, Child>, Child>;
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
      if (!Halfspace<HalfspaceMeta<Space>, Child, HalfspaceType::LH>(
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
      const auto half =
          Halfspace<HalfspaceMeta<Space>, Child, HalfspaceType::LH>(
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

    const auto [U_inv, event] = linalg::MatrixInverse<Mat>()(U);
    static_cast<void>(event);
    const auto A_new = A * U_inv;
    const auto b_new = b + A_new * trans;

    return Polyhedron<Meta, Child>(A_new, b_new, std::get<0>(this->children));
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
   * [old] homogeneous t embedding removed
   * C = { x | A x <= b, Cx = d }
   * K = { (x, t) | A x <= b t, Cx = d t, t >= 0 }
   *
   * -Ax >= -b... b t - Ax >= 0 in rplus
   *  t... in rplus
   *
   * [ -A b ] [ x; t ] >= 0
   * [ 0  1 ]
   *
   * -a0^T x0 + bt >= 0
   * ...
   * t >= 0
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

    const auto map = affine::AffineMap<decltype(A)>(T(-1) * A, b);
    const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, M>>();
    const auto cones = std::make_tuple(rplus);

    const auto new_builder =
        conic::CanonicalFormBuilder<conic::builder_dom_t<decltype(builder)>,
                                    decltype(map), decltype(child_cb),
                                    decltype(child_builder.eq_map)>(
            map, child_cb, child_builder.eq_map);
    return new_builder.template make<Type>(cones);
  }

  /** @brief member variables in the polyhedron */
  Mat A;   //< set of finite number of halfspaces
  Bias b;  //< codomain vector describing halfspace inequalities
};

template <typename Mat, typename Bias, typename Child>
Polyhedron(const Mat&, const Bias&, const Child&)
    -> Polyhedron<
        PolyhedronMeta<
            dom_space_t<meta_codom_t<atom_meta_t<Child>>>,
            linalg::Matrix<dom_type_t<meta_codom_t<atom_meta_t<Child>>>, 1,
                           linalg::mat_rows_v<Mat>>>,
        const Child>;

}  // namespace convex
