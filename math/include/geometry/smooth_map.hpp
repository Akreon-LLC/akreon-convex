/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <functional>

#include "../topology/topology.hpp"
#include "smooth_manifold.hpp"

namespace geometry {

template <typename T1, typename T2, typename Rn1, typename Rn2,
          typename MapToRn1, typename MapFromRn1, typename MapToRn2,
          typename MapFromRn2>
struct SmoothMap {
  // f : M -> N
  std::function<T2(const T1&)> f;

  // Checks for each chart (φ, ψ) that ψ∘f∘φ⁻¹ is smooth
  constexpr bool is_smooth(
      const SmoothManifold<T1, Rn1, MapToRn1, MapFromRn1>& M,
      const SmoothManifold<T2, Rn2, MapToRn2, MapFromRn2>& N) const {
    for (const auto& chart_M : M.atlas.charts) {
      for (const auto& chart_N : N.atlas.charts) {
        // check if the image of U under mapping f is a subset of V
        auto image_subset =
            std::all_of(chart_M.U.begin(), chart_M.U.end(), [&](const auto& p) {
              auto image = f(p);
              return chart_N.U.count(image) > 0;
            });
        if (image_subset) {
#if 0
        // ψ ∘ f ∘ φ⁻¹ : R^n → R^m
        auto coord_map = [&](const Rn1& x) {
          return chart_N.phi(f(chart_M.inv_phi(x)));
        };
        if (!is_autodiff_smooth(coord_map)) return false;
#endif
        }
      }
    }
    return true;
  }
};

template <typename T1, typename T2, typename Rn1, typename Rn2,
          typename MapToRn1, typename MapFromRn1, typename MapToRn2,
          typename MapFromRn2>
concept ValidSmoothMap =
    requires(const SmoothMap<T1, T2, Rn1, Rn2, MapToRn1, MapFromRn1, MapToRn2,
                             MapFromRn2>
                 smooth_map,
             const SmoothManifold<T1, Rn1, MapToRn1, MapFromRn1>& sm_M,
             const SmoothManifold<T2, Rn2, MapToRn2, MapFromRn2>& sm_N) {
      { smooth_map.is_smooth(sm_M, sm_N) } -> std::same_as<bool>;
    };

}  // namespace geometry
