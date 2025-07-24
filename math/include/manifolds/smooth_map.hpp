/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <functional>

#include "atlas.hpp"

namespace smooth_map {

template <typename T1, typename T2, typename Rn1, typename Rn2,
          typename MapToRn1, typename MapFromRn1, typename MapToRn2,
          typename MapFromRn2>
struct SmoothMap {
  // f : M -> N
  std::function<T2(const T1&)> f;

  // Checks for each chart (φ, ψ) that ψ∘f∘φ⁻¹ is smooth
  bool is_smooth(
      const atlas::Atlas<T1, Rn1, MapToRn1, MapFromRn1>& atlas_M,
      const atlas::Atlas<T2, Rn2, MapToRn2, MapFromRn2>& atlas_N) const;
};

template <typename T1, typename T2, typename Rn1, typename Rn2,
          typename MapToRn1, typename MapFromRn1, typename MapToRn2,
          typename MapFromRn2>
concept ValidSmoothMap =
    requires(const SmoothMap<T1, T2, Rn1, Rn2, MapToRn1, MapFromRn1, MapToRn2,
                             MapFromRn2>
                 smooth_map,
             const atlas::Atlas<T1, Rn1, MapToRn1, MapFromRn1>& atlas_M,
             const atlas::Atlas<T2, Rn2, MapToRn2, MapFromRn2>& atlas_N) {
      { smooth_map.is_smooth(atlas_M, atlas_N) } -> std::same_as<bool>;
    };

}  // namespace smooth_map
