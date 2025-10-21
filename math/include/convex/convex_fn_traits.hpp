/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set_traits.hpp"

namespace convex {

/// @brief enum for function type
enum struct FnTag { Convex = 0, Concave };

/// @brief forward declaration - convex fn
template <ConvexSetLike Space, typename Derived>
class ConvexFn;

/// @brief forward declaration - affine fn
template <linalg::RnSpace Space, FnTag Tag>
class AffineFn;

/// @brief forward declaration - indicator fn
template <ConvexSetLike Space, FnTag Tag>
class IndicatorFn;

/// @brief forward declaration - max fn
template <linalg::RnSpace Space, FnTag Tag>
class MaxFn;

/// @brief forward declaration - norm fn
template <linalg::RnSpace Space, linalg::NormLike Norm, FnTag Tag>
  requires(linalg::space_dim_v<Space> == linalg::norm_size_v<Norm>)
class NormFn;

/// @brief forward declaration - quad over linear fn
template <linalg::RnSpace Space, FnTag Tag>
  requires(linalg::space_dim_v<Space> >= 2)
class QuadOverLinearFn;

/// @brief fn traits
template <class Fn>
struct fn_traits;

/// @brief derived fn type traits - affine fn
template <linalg::RnSpace Space, FnTag Tag>
struct fn_traits<AffineFn<Space, Tag>> {
  using Domain = Space;
  static constexpr FnTag FuncTag = Tag;
};

/// @brief derived fn type traits - indicator fn
template <ConvexSetLike Space, FnTag Tag>
struct fn_traits<IndicatorFn<Space, Tag>> {
  using Domain = Space;
  static constexpr FnTag FuncTag = Tag;
};

/// @brief derived fn type traits - max fn
template <linalg::RnSpace Space, FnTag Tag>
struct fn_traits<MaxFn<Space, Tag>> {
  using Domain = Space;
  static constexpr FnTag FuncTag = Tag;
};

/// @brief derived fn type traits - norm fn
template <linalg::RnSpace Space, linalg::NormLike Norm, FnTag Tag>
struct fn_traits<NormFn<Space, Norm, Tag>> {
  using Domain = Space;
  static constexpr FnTag FuncTag = Tag;
};

/// @brief derived fn type traits - quad/linear fn
template <linalg::RnSpace Space, FnTag Tag>
struct fn_traits<QuadOverLinearFn<Space, Tag>> {
  using Domain = Space;
  static constexpr FnTag FuncTag = Tag;
};

/// @brief helpers to extract domain & function type
template <class Fn>
using fn_domain_t = typename fn_traits<linalg::remove_cvref_t<Fn>>::Domain;

template <class Fn>
inline constexpr FnTag fn_tag_v =
    fn_traits<linalg::remove_cvref_t<Fn>>::FuncTag;

/// @brief convex fn concept
template <class Fn>
concept ConvexFnLike = ConvexSetLike<fn_domain_t<Fn>>;

}  // namespace convex
