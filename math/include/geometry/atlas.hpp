/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <concepts>
#include <set>
#include <vector>

#include "chart.hpp"

namespace atlas {

template <typename T, typename Rn, typename MapToRn, typename MapFromRn>
  requires chart::ValidChart<T, Rn, MapToRn, MapFromRn>
struct Atlas {
  using ChartT = chart::Chart<T, Rn, MapToRn, MapFromRn>;
  std::vector<ChartT> charts;

  inline bool covers(const topology::Topology<T>& M) const {
    topology::Set<T> union_of_domains;
    for (const auto& chart : charts)
      union_of_domains.insert(chart.U.begin(), chart.U.end());

    return M.contains(union_of_domains);
  }

  inline bool is_compatible() const {
    auto is_close = [](const Rn& a, const Rn& b, double eps = 1e-6) {
      if (a.size() != b.size()) return false;
      for (size_t i = 0; i < (size_t)a.size(); ++i)
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
};

template <typename T, typename Rn, typename MapToRn, typename MapFromRn>
concept ValidAtlas = requires(const Atlas<T, Rn, MapToRn, MapFromRn>& atlas,
                              const topology::Topology<T>& M) {
  requires topology::TopologicalManifold<T>;  // M must satisfy the manifold
                                              // concept
  { atlas.covers(M) } -> std::same_as<bool>;
};

template <typename T, typename Rn, typename MapToRn, typename MapFromRn>
concept SmoothAtlas = ValidAtlas<T, Rn, MapToRn, MapFromRn> &&
                      requires(const Atlas<T, Rn, MapToRn, MapFromRn>& atlas) {
                        { atlas.is_compatible() } -> std::same_as<bool>;
                      };

}  // namespace atlas
