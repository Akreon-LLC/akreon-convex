/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include "math.hpp"

namespace chart {
template struct Chart<Real, RealN, Phi, InvPhi>;
}  // namespace chart

namespace atlas {
template struct Atlas<chart::Real, chart::RealN, chart::Phi, chart::InvPhi>;
} // namespace atlas

namespace smooth_manifold {
template struct SmoothManifold<chart::Real, chart::RealN, chart::Phi,
                               chart::InvPhi>;
} // namespace smooth_manifold

namespace smooth_map {
template struct SmoothMap<chart::Real, chart::Real, chart::RealN, chart::RealN,
                          chart::Phi, chart::InvPhi, chart::Phi, chart::InvPhi>;
} // namespace smooth_map
