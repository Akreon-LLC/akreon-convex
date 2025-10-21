/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set_traits.hpp"

namespace convex {

/// @brief forward declaration - convex fn
template <ConvexSetLike Space, typename Derived>
class ConvexFn;

/// @brief forward declaration - affine fn
template <linalg::RnSpace Space>
class AffineFn;

/// @brief forward declaration - indicator fn
template <ConvexSetLike Space>
class IndicatorFn;

/// @brief forward declaration - max fn
template <linalg::RnSpace Space>
class MaxFn;

/// @brief forward declaration - norm fn
template <linalg::RnSpace Space, linalg::NormLike Norm>
  requires(linalg::space_dim_v<Space> == linalg::norm_size_v<Norm>)
class NormFn;

/// @brief forward declaration - quad over linear fn
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> >= 2)
class QuadOverLinearFn;

/// @brief fn traits
template <class Fn>
struct fn_traits;

/// @brief derived fn type traits - convex fn
template <ConvexSetLike Space, typename Derived>
struct fn_traits<ConvexFn<Space, Derived>> {
  using Domain = Space;
};

/// @brief derived fn type traits - affine fn
template <linalg::RnSpace Space>
struct fn_traits<AffineFn<Space>> {
  using Domain = Space;
};

/// @brief derived fn type traits - indicator fn
template <ConvexSetLike Space>
struct fn_traits<IndicatorFn<Space>> {
  using Domain = Space;
};

/// @brief derived fn type traits - max fn
template <linalg::RnSpace Space>
struct fn_traits<MaxFn<Space>> {
  using Domain = Space;
};

/// @brief derived fn type traits - norm fn
template <linalg::RnSpace Space, linalg::NormLike Norm>
struct fn_traits<NormFn<Space, Norm>> {
  using Domain = Space;
};

/// @brief derived fn type traits - quad/linear fn
template <linalg::RnSpace Space>
struct fn_traits<QuadOverLinearFn<Space>> {
  using Domain = Space;
};

template <class Fn>
using fn_domain_t = typename fn_traits<linalg::remove_cvref_t<Fn>>::Domain;

/// @brief convex fn concept
template <class Fn>
concept ConvexFnLike = ConvexSetLike<fn_domain_t<Fn>>;

}  // namespace convex
