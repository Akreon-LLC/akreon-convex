/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_traits.hpp"

namespace convex {

/// @brief forward declaration of cone view
template <ConvexConeLike Cone>
class ConeView;

/// @brief cone traits
template <ConvexConeLike Cone>
struct cone_traits;

/// @brief derived cone type traits - hyperplane
template <linalg::VectorSpace Space>
struct cone_traits<Hyperplane<Space, ViewTag::ConvexCone>> {
  using ConeT = Hyperplane<Space, ViewTag::ConvexCone>;
  static constexpr bool is_closed(const ConeT& /*cone*/) { return true; }
  static constexpr bool is_solid(const ConeT& cone) {
    return cone.is_well_formed() && linalg::space_dim_v<Space> > 0;
  }
  static constexpr bool is_pointed(const ConeT& /*cone*/) { return false; }
};

/// @brief derived cone type traits - halfspace
template <linalg::VectorSpace Space, HalfspaceType Type>
struct cone_traits<Halfspace<Space, Type, ViewTag::ConvexCone>> {
  using ConeT = Halfspace<Space, Type, ViewTag::ConvexCone>;
  static constexpr bool is_closed(const ConeT& /*cone*/) {
    return Type == HalfspaceType::CLH || Type == HalfspaceType::CUH;
  }
  static constexpr bool is_solid(const ConeT& cone) {
    return cone.is_well_formed() && linalg::space_dim_v<Space> > 0;
  }
  static constexpr bool is_pointed(const ConeT& /*cone*/) { return false; }
};

/// @brief derived cone type traits - ellipsoid
template <linalg::VectorSpace Space, EllipsoidType Type>
struct cone_traits<Ellipsoid<Space, Type, ViewTag::ConvexCone>> {
  using ConeT = Ellipsoid<Space, Type, ViewTag::ConvexCone>;
  static constexpr bool is_closed(const ConeT& /*cone*/) {
    return Type == EllipsoidType::CE;
  }
  static constexpr bool is_solid(const ConeT& cone) {
    return cone.is_well_formed() && linalg::space_dim_v<Space> > 0;
  }
  /**
   * @brief pointed test (x,t) \in K & (-x, -t) \in -K
   *
   * (i) (x - ct)^T P^{-1} (x - ct) <= t^2
   * (ii) (-x + ct)^T P{-1} (-x + ct) <= (-t)^2 = t^2
   *
   * both inequalities are bounded by t^2 & (x - ct)^2 \equiv (-x + ct)^2 so
   * their relative magnitudes are the same (scaled by same P)
   *
   * -t < 0 and hence not in -K
   *
   *  K \cap -K = {0}
   */
  static constexpr bool is_pointed(const ConeT& /*cone*/) { return true; }
};

/// @brief derived cone type traits - norm ball
template <linalg::VectorSpace Space, NormConeType Type>
struct cone_traits<NormCone<Space, Type, ViewTag::ConvexCone>> {
  using ConeT = NormCone<Space, Type, ViewTag::ConvexCone>;
  static constexpr bool is_closed(const ConeT& /*cone*/) {
    return Type == NormConeType::L2CC || Type == NormConeType::L1CC ||
           Type == NormConeType::LinfCC;
  }
  static constexpr bool is_solid(const ConeT& cone) {
    return cone.is_well_formed() && linalg::space_dim_v<Space> > 0;
  }
  /**
   * @brief pointed test (x,t) \in K & (-x, -t) \in -K
   *
   * (i) { (x, t) | ||x|| <= t }
   * (ii) { (-x, -t) | ||-x|| <= -t }
   *
   * -t < 0 thus not in -K
   *
   * L2 case : x \in K, -x \in -K but -t invalid so K \cap -K = {0}
   * L1 case : similar result
   * Linf case : similar result
   */
  static constexpr bool is_pointed(const ConeT& /*cone*/) { return true; }
};

}  // namespace convex
