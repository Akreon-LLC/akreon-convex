/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include "math.hpp"

#if 0
namespace chart {
using M = Rn;
template struct Chart<M, Rn, Phi<M>, InvPhi<M>>;
}  // namespace chart

namespace atlas {
template struct Atlas<chart::R, chart::Rn, chart::Phi<chart::M>, chart::InvPhi<chart::M>>;
}  // namespace atlas

namespace smooth_manifold {
template struct SmoothManifold<chart::Real, chart::RealN, chart::Phi,
                               chart::InvPhi>;
} // namespace smooth_manifold

namespace smooth_map {
template struct SmoothMap<chart::Real, chart::Real, chart::RealN, chart::RealN,
                          chart::Phi, chart::InvPhi, chart::Phi, chart::InvPhi>;
} // namespace smooth_map
#endif
