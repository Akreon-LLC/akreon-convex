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

#include "convex/cone_block_builder.hpp"
#include "convex/cones/soc.hpp"
#include "convex/convex_traits.hpp"

namespace convex {

/// @brief alias for domain on inv pos function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using InvPosDom = ConvexDomain<Space, SignType::POS>;

/// @brief alias for codomain on inv pos function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using InvPosCodom = ConvexDomain<Space, SignType::POS>;

/// @brief alias for concave codomain on inv pos function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using InvPosConcaveCodom = ConvexDomain<Space, SignType::NEG>;

/// @brief alias for meta for inv pos function
template <linalg::RnSpace Space>
using InvPosMeta =
    ConvexAtomMeta<InvPosDom<Space>, InvPosCodom<Space>, CurvatureType::CONVEX>;

/// @brief alias for meta for inv pos function
template <linalg::RnSpace Space>
using InvPosConcaveMeta =
    ConvexAtomMeta<InvPosDom<Space>, InvPosConcaveCodom<Space>,
                   CurvatureType::CONCAVE>;

/**
 * @brief inv pos function
 *
 * f(x) = 1/x, x in R_++
 */
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct InvPos : Atom<Meta, InvPos<Meta, Child>, Child> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = Atom<Meta, InvPos<Meta, Child>, Child>;
  using ConvexBase::N;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /// @brief alias to extract space type
  using Space = dom_space_t<meta_dom_t<Meta>>;

  /// @brief curvature tag
  static constexpr CurvatureType Tag = meta_curv_v<Meta>;

  /**
   * @brief CTOR
   *
   * @param child
   */
  constexpr explicit InvPos(const Child& child_in)
      : ConvexBase(child_in),
        lift_map(affine::make_map_id<LiftedVarMat<T>>()) {}

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit InvPos(const Child& child_in,
                            const affine::AffineMap<LiftedVarMat<T>>& map_in)
      : ConvexBase(child_in), lift_map(map_in) {}

  /**
   * @brief mono
   *
   * @return monotonicity
   */
  constexpr MonotonicityType monotonicity() const {
    if constexpr (Tag == CurvatureType::CONVEX)
      return MonotonicityType::NONINC;
    else
      return MonotonicityType::NONDEC;
  }

  /**
   * @brief eval via operator overload
   *
   * @param point x
   *
   * @return result
   */
  constexpr T operator()(const PointT& x) const { return eval(x); }

  /**
   * @brief eval impl
   *
   * @param x \in R_++
   *
   * @return res \in R_++ or R_-- (depending on convex or concave)
   */
  constexpr T eval(const PointT& x) const {
    const auto alpha = lift_map.linear_part()(0)(0);
    const auto beta = lift_map.translation()(0);
    AK_ASSERT(x(0) > T(0) && "Invalid domain for 1/x");

    if constexpr (Tag == CurvatureType::CONVEX) {
      return alpha / x(0) + beta;
    } else {
      return T(-1) * alpha / x(0) + beta;
    }
  }

  /**
   * @brief mul
   *
   * @return abs fn (updated map)
   */
  template <T Scalar>
  constexpr auto mul() const {
    if constexpr (Scalar < T(0)) {
      const auto alpha = T(-1) * Scalar;
      affine::AffineMap<LiftedVarMat<T>> new_map(alpha * lift_map.linear_part(),
                                                 lift_map.translation());
      if constexpr (Tag == CurvatureType::CONCAVE) {
        return InvPos<InvPosMeta<Space>, Child>(std::get<0>(this->children),
                                                new_map);
      } else {
        return InvPos<InvPosConcaveMeta<Space>, Child>(
            std::get<0>(this->children), new_map);
      }
    } else {
      affine::AffineMap<LiftedVarMat<T>> new_map(
          Scalar * lift_map.linear_part(), lift_map.translation());
      return Exp<Meta, Child>(std::get<0>(this->children), new_map);
    }
  }

  /**
   * @brief add
   *
   * @param scalar
   *
   * @return abs fn (updated map)
   */
  constexpr auto add(const T& beta) const {
    LiftedVarBias<T> bias{beta + lift_map.translation()(0)};
    affine::AffineMap<LiftedVarMat<T>> new_map(lift_map.linear_part(), bias);
    return InvPos<Meta, Child>(std::get<0>(this->children), new_map);
  }

  /**
   * @brief canonicalize_impl
   *
   * (1) f(x) = 1/x
   * K = { (x, t) | 1/x <= t, x in R_++ implies t >= 0 }
   * K_rsoc = { (u, v, w) | ||u||_2^2 <= 2 v w, v > 0, w >= 0 }
   *
   * let u = sqrt(2), v = x, w = t
   * [ 0   0 ] [ x; t ] + [ sqrt(2); b; 0] -> (sqrt(2), a^T x + b, t)
   * [ a^T 0 ]
   * [ 0   1 ]
   *
   * (2) f(x) = -1/x
   * K = { (x, t) | 1/x <= -t, x in R_++ t in R_- }
   *
   * let u = sqrt(2), v = x, w = -t
   * [ 0    0 ] [ x; t ] + [ sqrt(2); b; 0 ]
   * [ a^T  0 ]
   * [ 0   -1 ]
   *
   * @param cone block builder
   *
   * @return cone block builder
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize_impl(const Builder& builder) const {
    const auto child_builder =
        std::get<0>(this->children)
            .template canonicalize<AtomType::FN>(builder);
    const auto child_coord = child_builder.coord_map;
    const auto child_cb = child_builder.cb;

    using AffineMatT = affine::affine_mat_t<decltype(child_coord)>;
    static constexpr size_t Dom = linalg::mat_cols_v<AffineMatT>;
    static constexpr size_t Codom = linalg::mat_rows_v<AffineMatT>;
    static_assert(Codom == 1,
                  "input affine map from cone block has invalid codom dim for "
                  "scalar fn domain (i.e., != 1");
    linalg::Matrix<T, 3, Dom + 1> F;
    F = linalg::set_block<1, 2, 0, Dom>(F, child_coord.linear_part());
    linalg::Vector<T, 3> g;
    g(0) = linalg::constexpr_sqrt_safe(T(2));
    g(1) = child_coord.translation()(0);

    if constexpr (Tag == CurvatureType::CONVEX) {
      F(2)(Dom) = T(1);
    } else {
      F(2)(Dom) = T(-1);
    }

    const auto rsoc_coords = affine::AffineMap<decltype(F)>(F, g);
    const auto coord_map_up =
        conic::Rotated2SOC<decltype(rsoc_coords)>()(rsoc_coords);
    const auto new_builder =
        conic::ConeBlockBuilder<conic::builder_dom_t<decltype(builder)>,
                                decltype(coord_map_up), decltype(child_cb)>(
            coord_map_up, child_cb);
    const auto rsoc = conic::SecondOrderCone<linalg::Vector<T, 3>>();
    const auto cones = std::make_tuple(rsoc);
    return new_builder.template make<Type>(idx, lift_map, cones);
  }

  /** @brief member vars */
  affine::AffineMap<LiftedVarMat<T>> lift_map;
  const int idx{-1};
};

}  // namespace convex
