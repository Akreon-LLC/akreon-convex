/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include "geometry/smooth_map.hpp"

#include <algorithm>

namespace smooth_map {

template <typename T1, typename T2, typename Rn1, typename Rn2,
          typename MapToRn1, typename MapFromRn1, typename MapToRn2,
          typename MapFromRn2>
bool SmoothMap<T1, T2, Rn1, Rn2, MapToRn1, MapFromRn1, MapToRn2, MapFromRn2>::
    is_smooth(const smooth_manifold::SmoothManifold<T1, Rn1, MapToRn1,
                                                    MapFromRn1>& sm_M,
              const smooth_manifold::SmoothManifold<T2, Rn2, MapToRn2,
                                                    MapFromRn2>& sm_N) const {
  for (const auto& chart_M : sm_M.atlas.charts) {
    for (const auto& chart_N : sm_N.atlas.charts) {
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

template struct SmoothMap<chart::Real, chart::Real, chart::RealN, chart::RealN,
                          chart::Phi<chart::Real>, chart::InvPhi<chart::Real>,
                          chart::Phi<chart::Real>, chart::InvPhi<chart::Real>>;

}  // namespace smooth_map
