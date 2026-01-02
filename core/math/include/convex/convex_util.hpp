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
#include "convex/cone_traits.hpp"
#include "convex/convex_traits.hpp"

namespace convex {

/// @brief cone map builder functor
template <linalg::RnSpace DecVarDom, affine::AffineMapLike CoordMap,
          conic::ConeBlockLike CB>
struct ConeBlockBuilder {
  /// @brief alias extraction of underlying types
  using CoordMapT = affine::affine_mat_t<CoordMap>;
  using T = linalg::mat_type_t<CoordMapT>;

  /// @brief N dimension spanning dec var dom R^n
  static constexpr size_t N = linalg::space_dim_v<DecVarDom>;

  /**
   * @brief CTOR
   *
   * @param linear selector
   * @param cone block
   */
  constexpr explicit ConeBlockBuilder(const CoordMap& coord_in, const CB& cb_in)
      : coord_map(coord_in), cb(cb_in) {}

  /**
   * @brief update cone map
   *
   * @param selector idx from start aux col
   * @param emitted cones from child
   *
   * @return new cone block builder
   */
  template <conic::ConeLike... Cs>
  constexpr auto operator()(const size_t& idx,
                            const std::tuple<Cs...>& cones) const {
    const auto& cone_map = cb.map;
    using ConeMapT = affine::affine_mat_t<decltype(cone_map)>;

    static_assert(std::same_as<T, linalg::mat_type_t<ConeMapT>>,
                  "type mismatch between cone map and coord map");

    static constexpr size_t CoordDom = linalg::mat_cols_v<CoordMapT>;
    static constexpr size_t CoordCodom = linalg::mat_rows_v<CoordMapT>;

    static constexpr size_t ConeDom = linalg::mat_cols_v<ConeMapT>;
    static constexpr size_t ConeCodom = linalg::mat_rows_v<ConeMapT>;

    static constexpr size_t ConeAux = ConeDom == 0 ? 0 : ConeDom - N;
    static constexpr size_t CoordAux = CoordDom - N;
    static constexpr size_t CoordAuxStartCol = ConeDom == 0 ? N : ConeDom;
    static constexpr size_t CoordAuxEndCol = CoordAuxStartCol + CoordAux;

    /// (dakre) F cols: [ dec vars | cone aux | coord aux ]
    static constexpr size_t FullDom = N + ConeAux + CoordAux;
    static constexpr size_t FullCodom = ConeCodom + CoordCodom;

    linalg::Matrix<T, FullCodom, FullDom> F;

    /// (0) fill upper left with prev cone map [ dec var | cone aux ]
    if constexpr (ConeDom != 0) {
      F = linalg::set_block<0, ConeCodom, 0, ConeDom>(F,
                                                      cone_map.linear_part());
    }

    /// (1) fill bottom left from coord map
    const auto A = coord_map.linear_part();
    const auto dec_spanning_blk = linalg::get_block<0, CoordCodom, 0, N>(A);
    const auto coord_aux_blk = linalg::get_block<0, CoordCodom, N, CoordDom>(A);
    F = linalg::set_block<ConeCodom, FullCodom, 0, N>(F, dec_spanning_blk);

    /// (2) fill bottom right with new aux vars from latest cone
    F = linalg::set_block<ConeCodom, FullCodom, CoordAuxStartCol,
                          CoordAuxEndCol>(F, coord_aux_blk);

    linalg::Vector<T, FullCodom> g;
    for (size_t i = 0; i < ConeCodom; ++i) g[i] = cone_map.translation()(i);
    for (size_t i = 0; i < CoordCodom; ++i)
      g[i + ConeCodom] = coord_map.translation()(i);

    linalg::Vector<T, FullDom> sel;
    if (CoordAuxStartCol + idx >= FullDom)
      throw std::domain_error("Selecting aux var that is out of bounds");

    sel[CoordAuxStartCol + idx] = T(1);
    const auto sel_row = linalg::as_row_vector(sel);
    const auto new_coord_map =
        affine::AffineMap<decltype(sel_row)>(sel_row, linalg::Vector<T, 1>{});
    const auto new_map = affine::AffineMap<decltype(F)>(F, g);
    const auto new_cb =
        conic::ConeBlock(new_map, std::tuple_cat(cb.cones, cones));
    return ConeBlockBuilder<DecVarDom, decltype(new_coord_map),
                            decltype(new_cb)>(new_coord_map, new_cb);
  }

  CoordMap coord_map;
  CB cb;
};

/* @brief create empty cone block
 *
 * @return cone block
 */
template <linalg::FieldLike T>
constexpr auto create_empty_cblk() {
  constexpr auto empty = std::make_tuple();
  constexpr conic::ConeBlock cb(
      affine::AffineMap(linalg::Matrix<T, 0, 0>{}, linalg::Vector<T, 0>{}),
      empty);
  return cb;
}

template <linalg::RnSpace DecVarDom>
constexpr auto create_empty_cblk_builder() {
  using T = linalg::space_type_t<DecVarDom>;
  const auto cblk = create_empty_cblk<T>();
  return ConeBlockBuilder<DecVarDom, decltype(cblk.map), decltype(cblk)>(
      cblk.map, cblk);
}

}  // namespace convex
