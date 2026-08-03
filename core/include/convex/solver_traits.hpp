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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "config/solver_config.hpp"
#include "convex/canonical_trace.hpp"

namespace convex {

/// @brief forward declaration of pd result
template <linalg::VectorContainer Var, linalg::VectorContainer ConeVar,
          linalg::VectorContainer NuVar, typename CanonicalTrace,
          typename SolverTrace>
struct PrimalDualResult;

} // namespace convex
