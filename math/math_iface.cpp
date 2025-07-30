/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include "math.hpp"

namespace geometry {

using Real = double;
template <size_t N>
using R =
    topology::EuclideanTopology<algebra::VectorSpace<Real, Eigen::Dynamic>, N>;

template <typename M, size_t N>
using Phi = std::function<R<N>(const M&)>;

template <typename M, size_t N>
using InvPhi = std::function<R<N>(const R<N>&)>;

#if 0
constexpr size_t N = 128;
using M = R<N>;
template struct Chart<M, R<N>, Phi<M, N>, InvPhi<M, N>>;

template struct Atlas<chart::R, chart::Rn, chart::Phi<chart::M>, chart::InvPhi<chart::M>>;

template struct SmoothManifold<chart::Real, chart::RealN, chart::Phi,
                               chart::InvPhi>;

template struct SmoothMap<chart::Real, chart::Real, chart::RealN, chart::RealN,
                          chart::Phi, chart::InvPhi, chart::Phi, chart::InvPhi>;
#endif
}  // namespace geometry
