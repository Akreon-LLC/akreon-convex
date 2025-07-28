/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <concepts>
#include <functional>
#include <set>
#include <vector>

#include "../algebra/vector_space.hpp"
#include "../topology/topology.hpp"

namespace chart {

using R = double;
using Rn = vector_space::VectorSpace<R, Eigen::Dynamic>;

template <typename M>
using Phi = std::function<Rn(const M&)>;

template <typename M>
using InvPhi = std::function<M(const Rn&)>;

template <typename M, typename Rn, typename MapToRn, typename MapFromRn>
  requires requires { typename M::value_type; }
struct Chart {
  using T = typename M::value_type;
  topology::Set<T> U;  // domain open subset U of the manifold M
  MapToRn phi;         // map M -> R^n
  MapFromRn inv_phi;   // inverse map R^n -> M
};

template <typename M, typename Rn, typename MapToRn, typename MapFromRn>
constexpr bool is_homeomorphism(const Chart<M, Rn, MapToRn, MapFromRn>& chart) {
  auto y = chart.phi(chart.U);           // U -> R^n
  auto U_back = chart.inv_phi(y);  // R^n -> U

  if (U_back != chart.U) return false;
  return true;
}

template <typename M, typename Rn, typename MapToRn, typename MapFromRn>
concept ValidChart =
    topology::TopologicalManifold<typename M::value_type> &&
    requires(const Chart<M, Rn, MapToRn, MapFromRn>& chart, const M& manifold) {
      // domain must be an open subset of M
      { manifold.contains(chart.U) } -> std::same_as<bool>;
      // phi must be a valid bijection with a continuous inverse
      { is_homeomorphism(chart) } -> std::same_as<bool>;
    };

}  // namespace chart
