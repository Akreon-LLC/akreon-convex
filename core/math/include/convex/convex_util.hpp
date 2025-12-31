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
   * @param selector idx from end (expects negative number)
   * @param emitted cones from child
   *
   * @return new cone block builder
   */
  template <conic::ConeLike... Cs>
  constexpr auto operator()(const int& ridx,
                            const std::tuple<Cs...>& cones) const {
    const auto& cone_map = cb.map;
    using ConeMapT = affine::affine_mat_t<decltype(cone_map)>;

    static_assert(std::same_as<T, linalg::mat_type_t<ConeMapT>>,
                  "type mismatch between cone map and coord map");
    static constexpr size_t SelDom = linalg::mat_cols_v<CoordMapT>;
    static constexpr size_t SelCodom = linalg::mat_rows_v<CoordMapT>;

    static constexpr size_t ConeDom = linalg::mat_cols_v<ConeMapT>;
    static constexpr size_t ConeCodom = linalg::mat_rows_v<ConeMapT>;

    static constexpr size_t FullDom =
        ConeDom == 0 ? SelDom : ConeDom + SelDom - N;
    static constexpr size_t FullCodom = SelCodom + ConeCodom;

    linalg::Matrix<T, FullCodom, FullDom> F;
    F = linalg::set_block<0, ConeCodom, 0, ConeDom>(F, cone_map.linear_part());

    const auto A = coord_map.linear_part();
    // const auto b = coord_map.translation();
    const auto dec_spanning_blk = linalg::get_block<0, SelCodom, 0, N>(A);
    const auto aux_spanning_blk = linalg::get_block<0, SelCodom, N, SelDom>(A);

    F = linalg::set_block<ConeCodom, FullCodom, 0, N>(F, dec_spanning_blk);
    F = linalg::set_block<ConeCodom, FullCodom, N, FullDom>(F,
                                                            aux_spanning_blk);

    linalg::Vector<T, FullCodom> g;
    for (size_t i = 0; i < ConeCodom; ++i) g[i] = cone_map.translation()(i);
    for (size_t i = 0; i < SelCodom; ++i)
      g[i + ConeCodom] = coord_map.translation()(i);

    linalg::Vector<T, FullDom> sel;
    sel[FullDom + ridx] = T(1);
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
