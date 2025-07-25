/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include "manifolds/atlas.hpp"

#include <algorithm>

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

template <typename T, typename Rn, typename MapToR, typename MapFromR>
  requires chart::ValidChart<T, Rn, MapToR, MapFromR>
bool Atlas<T, Rn, MapToR, MapFromR>::is_compatible() const {
  auto is_close = [](const Rn& a, const Rn& b, double eps = 1e-6) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
      if (std::abs(a[i] - b[i]) > eps) return false;
    return true;
  };

  // p \in U_i \intersection U_j 
  // Φ_i ∘ Φ_j^{-1} ∘ Φ_i(p) => validation for Φ_i ∘ Φ_j^{-1} transition map
  for (size_t i = 0; i < charts.size(); ++i) {
    for (size_t j = i + 1; j < charts.size(); ++j) {
      topology::Set<T> inter;
      std::set_intersection(charts[i].U.begin(), charts[i].U.end(),
                            charts[j].U.begin(), charts[j].U.end(),
                            std::inserter(inter, inter.begin()));
      if (inter.empty()) continue;

      // p \in U_i \intersection U_j
      for (const auto& p : inter) {
        Rn phi_i_image_on_p = charts[i].phi(p);  // intersection -> R^n
        T phi_j_preimage = charts[j].inv_phi(phi_i_image_on_p);  // R^n -> U_j
        Rn phi_i_image_on_j = charts[i].phi(phi_j_preimage);     // U_j -> R^n

        if (!is_close(phi_i_image_on_j, phi_i_image_on_p)) return false;
      }
    }
  }
  return true;
}

template struct Atlas<topology::Real, topology::RealN,
                      chart::Phi<topology::Real>,
                      chart::InvPhi<topology::Real>>;

}  // namespace atlas
