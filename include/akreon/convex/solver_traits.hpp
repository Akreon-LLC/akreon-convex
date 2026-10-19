/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "akreon/convex/solver_config.hpp"
#include "akreon/convex/canonical_trace.hpp"

namespace convex {

/// @brief forward declaration of pd result
template <linalg::VectorContainer Var, linalg::VectorContainer ConeVar,
          linalg::VectorContainer NuVar, typename CanonicalTrace,
          typename SolverTrace>
struct PrimalDualResult;

} // namespace convex
