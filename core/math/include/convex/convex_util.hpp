/*
 * Copyright © 2025 Akreon LLC. All rights reserved.
 *
 * Proprietary and Confidential.
 * This source code is the property of Akreon LLC and constitutes a trade secret
 * protected by U.S. and international law.
 *
 * Unauthorized copying, modification, distribution, or disclosure of this file,
 * via any medium, is strictly prohibited without express written permission
 * from Akreon LLC.
 */

#pragma once

#include "affine/affine_map.hpp"
#include "convex/convex_traits.hpp"

namespace convex {

/// @brief cone map builder functor
template <linalg::RnSpace DecVarDom>
struct ConeMapBuilder {
  /// @brief N dimension spanning dec var dom R^n
  static constexpr size_t N = linalg::space_dim_v<DecVarDom>;

  /**
   * @brief update cone map
   *
   * @param inner coord map
   * @param full map
   * @param selector idx from end (expects negative number)
   *
   * @return (linear selector, full affine map)
   */
  template <affine::AffineMapLike CoordMap, affine::AffineMapLike ConeMap>
  constexpr auto operator()(const CoordMap& coord_map, const ConeMap& cone_map,
                            const int& ridx) const {
    using CoordMapT = affine::affine_mat_t<CoordMap>;
    using ConeMapT = affine::affine_mat_t<ConeMap>;
    using T = linalg::mat_type_t<CoordMapT>;
    static_assert(std::same_as<T, linalg::mat_type_t<ConeMapT>>,
                  "type mismatch between cone map and coord map");
    static constexpr size_t CoordDom = linalg::mat_cols_v<CoordMapT>;
    static constexpr size_t CoordCodom = linalg::mat_rows_v<CoordMapT>;

    static constexpr size_t ConeDom = linalg::mat_cols_v<ConeMapT>;
    static constexpr size_t ConeCodom = linalg::mat_rows_v<ConeMapT>;

    static constexpr size_t FullDom =
        ConeDom == 0 ? CoordDom : ConeDom + CoordDom - N;
    static constexpr size_t FullCodom = CoordCodom + ConeCodom;

    linalg::Matrix<T, FullCodom, FullDom> F;
    F = linalg::set_block<0, ConeCodom, 0, ConeDom>(F, cone_map.linear_part());

    const auto A = coord_map.linear_part();
    const auto dec_spanning_blk = linalg::get_block<0, CoordCodom, 0, N>(A);
    const auto aux_spanning_blk =
        linalg::get_block<0, CoordCodom, N, CoordDom>(A);

    F = linalg::set_block<ConeCodom, FullCodom, 0, N>(F, dec_spanning_blk);
    F = linalg::set_block<ConeCodom, FullCodom, N, FullDom>(F,
                                                            aux_spanning_blk);

    linalg::Vector<T, FullCodom> g;
    for (size_t i = 0; i < ConeCodom; ++i) g[i] = cone_map.translation()(i);
    for (size_t i = 0; i < CoordCodom; ++i)
      g[i + ConeCodom] = coord_map.translation()(i);

    linalg::Vector<T, FullDom> sel;
    sel[FullDom + ridx] = T(1);
    const auto sel_row = linalg::as_row_vector(sel);
    return std::tuple(
        affine::AffineMap<decltype(sel_row)>(sel_row, linalg::Vector<T, 1>{}),
        affine::AffineMap<decltype(F)>(F, g));
  }
};

}  // namespace convex
