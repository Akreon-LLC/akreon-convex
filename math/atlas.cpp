/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include "manifolds/atlas.h"

namespace atlas {

template <typename T, typename Rn, typename MapToR, typename MapFromR>
  requires chart::ValidChart<T, Rn, MapToR, MapFromR>
bool Atlas<T, Rn, MapToR, MapFromR>::covers(
    const topology::Topology<T>& M) const {
  topology::Set<T> union_of_domains;
  for (const auto& chart : charts)
    union_of_domains.insert(chart.U.begin(), chart.U.end());

  return M.contains(union_of_domains);
}

template struct Atlas<topology::Real, topology::RealN, chart::Phi, chart::InvPhi>;

}  // namespace atlas
