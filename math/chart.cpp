/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include "manifolds/chart.h"

namespace chart {

template <typename T, typename Rn, typename MapToR, typename MapFromR>
bool is_homeomorphism(const Chart<T, Rn, MapToR, MapFromR>& chart) {
  for (const T& x : chart.U) {
    auto y = chart.phi(x);           // x \in U -> R^n
    auto x_back = chart.inv_phi(y);  // R^n -> x \in U

    if (x_back != x) return false;
  }
  return true;
}

template struct Chart<topology::Real, topology::RealN, Phi, InvPhi>;

}  // namespace chart
