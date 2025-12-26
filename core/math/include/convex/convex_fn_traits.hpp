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
template <ConvexDomainLike Domain, typename Derived, ConvexAtomLike Child>
struct ConvexFn;

/// @brief forward declaration - max fn
template <ConvexDomainLike Domain, FnTag Tag>
class MaxFn;

/// @brief forward declaration - norm fn
template <ConvexDomainLike Domain, linalg::NormLike Norm, FnTag Tag>
  requires(dom_size_v<Domain> == linalg::norm_size_v<Norm>)
class NormFn;

/// @brief forward declaration - geo mean fn
template <ConvexDomainLike Domain, FnTag Tag>
class GeoMeanFn;

/// @brief forward declaration - exp fn
template <ConvexDomainLike Domain, FnTag Tag>
  requires(dom_size_v<Domain> == 1)
class ExpFn;

/// @brief forward declaration - log fn
template <ConvexDomainLike Domain, FnTag Tag>
  requires(dom_size_v<Domain> == 1)
class LogFn;

/// @brief forward declaration - square fn
template <ConvexDomainLike Domain, FnTag Tag>
  requires(dom_size_v<Domain> == 1)
class SquareFn;

/// @brief forward declaration - abs fn
template <ConvexDomainLike Domain, ConvexAtomLike Child, FnTag Tag>
  requires(dom_size_v<Domain> == 1)
class AbsFn;

/// @brief forward declaration - inv pos fn
template <ConvexDomainLike Domain, FnTag Tag>
  requires(dom_size_v<Domain> == 1)
class InvPosFn;

/// @brief forward declaration - log sum exp fn
template <ConvexDomainLike Domain, FnTag Tag>
class LogSumExpFn;

// (dakre) TODO currently unimplemented but forward dec stub here

/// @brief forward declaration - log det fn
template <ConvexDomainLike Domain, FnTag Tag>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<Domain>>)
class LogDetFn;

/// @brief forward declaration - harmonic mean fn
template <ConvexDomainLike Domain, FnTag Tag>
class HarmonicMeanFn;

/// @brief forward declaration - sigmoid fn
template <ConvexDomainLike Domain, FnTag Tag>
class SigmoidFn;

/// @brief forward declaration - huber fn
template <ConvexDomainLike Domain, FnTag Tag>
class HuberFn;

/// @brief forward declaration - kl divergence
template <ConvexDomainLike Domain, FnTag Tag>
class KLDivergenceFn;

/// @brief fn traits
template <class Fn>
struct convex_fn_traits : std::false_type {};

/// @brief derived fn type traits - max fn
template <ConvexDomainLike Dom>
struct convex_fn_traits<MaxFn<Dom, FnTag::Convex>> : std::true_type {
  using Domain = Dom;
};

/// @brief derived fn type traits - norm fn
template <ConvexDomainLike Dom, linalg::NormLike Norm>
struct convex_fn_traits<NormFn<Dom, Norm, FnTag::Convex>> : std::true_type {
  using Domain = Dom;
};

/// @brief derived fn type traits - log det fn
template <ConvexDomainLike Dom>
struct convex_fn_traits<GeoMeanFn<Dom, FnTag::Convex>> : std::true_type {
  using Domain = Dom;
};

/// @brief derived fn type traits - exp fn
template <ConvexDomainLike Dom>
struct convex_fn_traits<ExpFn<Dom, FnTag::Convex>> : std::true_type {
  using Domain = Dom;
};

/// @brief derived fn type traits - log fn
template <ConvexDomainLike Dom>
struct convex_fn_traits<LogFn<Dom, FnTag::Convex>> : std::true_type {
  using Domain = Dom;
};

/// @brief derived fn type traits - square fn
template <ConvexDomainLike Dom>
struct convex_fn_traits<SquareFn<Dom, FnTag::Convex>> : std::true_type {
  using Domain = Dom;
};

/// @brief derived fn type traits - abs fn
template <ConvexDomainLike Dom, ConvexAtomLike Child>
struct convex_fn_traits<AbsFn<Dom, Child, FnTag::Convex>> : std::true_type {
  using Domain = Dom;
};

/// @brief derived fn type traits - inv pos fn
template <ConvexDomainLike Dom>
struct convex_fn_traits<InvPosFn<Dom, FnTag::Convex>> : std::true_type {
  using Domain = Dom;
};

/// @brief derived fn type traits - log sum exp
template <ConvexDomainLike Dom>
struct convex_fn_traits<LogSumExpFn<Dom, FnTag::Convex>> : std::true_type {
  using Domain = Dom;
};

/// @brief derived fn type traits - log det fn
template <ConvexDomainLike Dom>
struct convex_fn_traits<LogDetFn<Dom, FnTag::Convex>> : std::true_type {
  using Domain = Dom;
};

/// @brief derived fn type traits - harmonic mean fn
template <ConvexDomainLike Dom>
struct convex_fn_traits<HarmonicMeanFn<Dom, FnTag::Convex>> : std::true_type {
  using Domain = Dom;
};

/// @brief derived fn type traits - sigmoid mean fn
template <ConvexDomainLike Dom>
struct convex_fn_traits<SigmoidFn<Dom, FnTag::Convex>> : std::true_type {
  using Domain = Dom;
};

/// @brief derived fn type traits - huber fn
template <ConvexDomainLike Dom>
struct convex_fn_traits<HuberFn<Dom, FnTag::Convex>> : std::true_type {
  using Domain = Dom;
};

/// @brief derived fn type traits - kl divergence fn
template <ConvexDomainLike Dom>
struct convex_fn_traits<KLDivergenceFn<Dom, FnTag::Convex>> : std::true_type {
  using Domain = Dom;
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
