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

#include "convex_set_traits.hpp"

namespace convex {

/// @brief tag dispatch for function type
enum struct FnTag { Convex = 0, Concave };

/// @brief forward declaration - convex fn
template <ConvexSetLike Space, typename Derived>
class ConvexFn;

/// @brief forward declaration - affine fn
template <linalg::RnSpace Space, FnTag Tag>
class AffineFn;

/// @brief forward declaration - indicator fn
template <ConvexSetLike Space>
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

/// @brief forward declaration - log sum exp fn
template <linalg::RnSpace Domain, FnTag Tag>
class LogSumExpFn;

/// @brief forward declaration - log det fn
template <linalg::RnSpace Domain, FnTag Tag>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<Domain>>)
class LogDetFn;

/// @brief forward declaration - geo mean fn
template <linalg::RnSpace Domain, FnTag Tag>
class GeoMeanFn;

/// @brief fn traits
template <class Fn>
struct convex_fn_traits : std::false_type {};

/// @brief derived fn type traits - affine fn
template <linalg::RnSpace Space>
struct convex_fn_traits<AffineFn<Space, FnTag::Convex>> : std::true_type {
  using Domain = Space;
};

/// @brief derived fn type traits - indicator fn
template <ConvexSetLike Space>
struct convex_fn_traits<IndicatorFn<Space>> : std::true_type {
  using Domain = Space;
};

/// @brief derived fn type traits - max fn
template <linalg::RnSpace Space>
struct convex_fn_traits<MaxFn<Space, FnTag::Convex>> : std::true_type {
  using Domain = Space;
};

/// @brief derived fn type traits - norm fn
template <linalg::RnSpace Space, linalg::NormLike Norm>
struct convex_fn_traits<NormFn<Space, Norm, FnTag::Convex>> : std::true_type {
  using Domain = Space;
};

/// @brief derived fn type traits - quad/linear fn
template <linalg::RnSpace Space>
struct convex_fn_traits<QuadOverLinearFn<Space, FnTag::Convex>>
    : std::true_type {
  using Domain = Space;
};

/// @brief derived fn type traits - log sum exp
template <linalg::RnSpace Space>
struct convex_fn_traits<LogSumExpFn<Space, FnTag::Convex>> : std::true_type {
  using Domain = Space;
};

/// @brief derived fn type traits - log det fn
template <linalg::RnSpace Space>
struct convex_fn_traits<LogDetFn<Space, FnTag::Convex>> : std::true_type {
  using Domain = Space;
};

/// @brief derived fn type traits - log det fn
template <linalg::RnSpace Space>
struct convex_fn_traits<GeoMeanFn<Space, FnTag::Convex>> : std::true_type {
  using Domain = Space;
};

/// @brief helpers to extract domain & function type
template <class Fn>
using convex_fn_domain_t =
    typename convex_fn_traits<linalg::remove_cvref_t<Fn>>::Domain;

/// @brief helper to extract convex convex_fn type
template <class Fn>
inline constexpr bool is_convex_fn_v =
    convex_fn_traits<linalg::remove_cvref_t<Fn>>{};

/// @brief convex fn concept
template <class Fn>
concept ConvexFnLike = is_convex_fn_v<Fn>;

/// @brief concave fn concept
template <class Fn>
concept ConcaveFnLike = !is_convex_fn_v<Fn>;

}  // namespace convex
