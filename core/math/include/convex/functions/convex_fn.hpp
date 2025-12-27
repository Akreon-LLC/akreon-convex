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

#include "convex/convex_fn_traits.hpp"
#include "convex/convex_util.hpp"

namespace convex {

/**
 * @brief base convex function CRTP
 *
 * Similar space constraints on R^n with convex set domain
 */
template <typename Meta, typename Derived, typename Child>
struct ConvexFn : public Atom<Meta, ConvexFn<Meta, Derived, Child>, Child> {
  /// @brief alias to extract type traits from Domain convex set
  using BaseT = Atom<Meta, ConvexFn<Meta, Derived, Child>, Child>;
  using typename BaseT::LinearMatT;
  using typename BaseT::PointT;
  using typename BaseT::T;
  using typename BaseT::TransT;

  /**
   * @brief CTOR
   *
   * @param domain
   * @param curvature
   * @param range
   * @param children
   */
  constexpr explicit ConvexFn(const Child& child_in,
                              const MonotonicityType& mono_in)
      : BaseT(std::make_tuple(child_in)), mono(mono_in) {}

  /**
   * @brief CTOR
   *
   * @param domain
   * @param curvature
   * @param range
   * @param children
   * @param affine map
   */
  constexpr explicit ConvexFn(const Child& child_in,
                              const affine::AffineMap<LinearMatT>& map_in,
                              const MonotonicityType& mono_in)
      : BaseT(std::make_tuple(child_in), map_in), mono(mono_in) {}

  /**
   * @brief operator overload for evaluation fn
   *
   * @param point x \in domain
   *
   * @return y \in R
   */
  constexpr std::optional<T> operator()(const PointT& x) const {
    return self().eval_impl(x);
  }

  /**
   * @brief eval
   *
   * F(x) -> y \in R
   *
   * @param point x \in domain
   *
   * @return y (scalar in R)
   */
  constexpr std::optional<T> eval(const PointT& x) const {
    return self().eval_impl(x);
  }

  /**
   * @brief negate f -> concave function or id
   *
   * @return negated convex fn
   */
  constexpr auto negate() const { return self().negate_impl(); }

  /**
   * @brief canonicalize
   *
   * f(x) -> epi / hypo -> homogenized cone
   *
   * @param cone map builder
   * @param coord map
   * @param cone block
   *
   * @return updated cone block
   */
  template <linalg::RnSpace DecVarDom, affine::AffineMapLike CoordMap,
            affine::AffineMapLike ConeMap, conic::ConeLike... Cones>
  constexpr auto canonicalize(
      const ConeMapBuilder<DecVarDom>& cmap_builder, const CoordMap& coord_map,
      const conic::ConeBlock<ConeMap, Cones...>& cone_block) const {
    return self().canonicalize_impl(cmap_builder, coord_map, cone_block);
  }

  /**
   * @brief self
   *
   * @return derived class instance
   */
  constexpr auto self() const { return static_cast<const Derived&>(*this); }

  /** @brief member vars */
  MonotonicityType mono;  ///< mono tag for convex function
};

}  // namespace convex
