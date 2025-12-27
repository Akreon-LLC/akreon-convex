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

/// @brief forward declaration - convex fn
template <typename Meta, typename Derived, typename Child>
struct ConvexFn;

/// @brief forward declaration - max fn
template <typename Meta>
class MaxFn;

/// @brief forward declaration - norm fn
template <typename Meta, linalg::NormLike Norm>
  requires(dom_size_v<dom_type_t<Meta>> == linalg::norm_size_v<Norm>)
class NormFn;

/// @brief forward declaration - geo mean fn
template <typename Meta>
class GeoMeanFn;

/// @brief forward declaration - exp fn
template <typename Meta>
  requires(dom_size_v<dom_type_t<Meta>> == 1)
class ExpFn;

/// @brief forward declaration - log fn
template <typename Meta>
  requires(dom_size_v<dom_type_t<Meta>> == 1)
class LogFn;

/// @brief forward declaration - square fn
template <typename Meta>
  requires(dom_size_v<dom_type_t<Meta>> == 1)
class SquareFn;

/// @brief forward declaration - abs fn
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
class AbsFn;

/// @brief forward declaration - inv pos fn
template <typename Meta>
  requires(dom_size_v<dom_type_t<Meta>> == 1)
class InvPosFn;

/// @brief forward declaration - log sum exp fn
template <typename Meta>
class LogSumExpFn;

// (dakre) TODO currently unimplemented but forward dec stub here

/// @brief forward declaration - log det fn
template <typename Meta>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<dom_type_t<Meta>>>)
class LogDetFn;

/// @brief forward declaration - harmonic mean fn
template <typename Meta>
class HarmonicMeanFn;

/// @brief forward declaration - sigmoid fn
template <typename Meta>
class SigmoidFn;

/// @brief forward declaration - huber fn
template <typename Meta>
class HuberFn;

/// @brief forward declaration - kl divergence
template <typename Meta>
class KLDivergenceFn;

}  // namespace convex
