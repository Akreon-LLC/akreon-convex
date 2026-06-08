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
#include "config/assert.hpp"
#include "convex/cone_traits.hpp"
#include "convex/convex_traits.hpp"

namespace conic {

/**
 * @brief Builder for conic canonical forms.
 *
 * The builder carries the current cone coordinate selector, emitted cone block,
 * and equality map while atoms append rows and auxiliary variables during
 * canonicalization.
 */
template <linalg::RnSpace DecVarDom, affine::AffineMapLike CoordMap,
          conic::ConeBlockLike CB, affine::AffineMapLike EqMap>
struct CanonicalFormBuilder {
  /// @brief Affine matrix type for the current cone-coordinate selector.
  using CoordMapT = affine::affine_mat_t<CoordMap>;
  /// @brief Scalar field used by the canonical form.
  using T = linalg::mat_type_t<CoordMapT>;

  /// @brief Number of original decision variables before auxiliary columns.
  static constexpr size_t N = linalg::space_dim_v<DecVarDom>;

  /**
   * @brief Construct a builder from existing canonical-form components.
   *
   * @param coord_in affine selector for the current atom coordinates
   * @param cb_in accumulated cone block
   * @param eq_in accumulated equality map
   */
  constexpr explicit CanonicalFormBuilder(const CoordMap& coord_in,
                                          const CB& cb_in, const EqMap& eq_in)
      : coord_map(coord_in), cb(cb_in), eq_map(eq_in) {}

  /**
   * @brief Append child canonical data and select a lifted auxiliary variable.
   *
   * @param idx negative index from the end of the updated coordinate domain
   * @param lift_map affine map that selects the lifted epigraph coordinate
   * @param cones emitted cones from the child atom
   *
   * @return builder with updated cone block, equality map, and coordinate map
   */
  template <convex::AtomType Type, affine::AffineMapLike LiftedMap,
            typename... Cs>
  constexpr auto make(const int& idx, const LiftedMap& lift_map,
                      const std::tuple<Cs...>& cones) const {
    const auto new_map = this->template update_map<Type>();
    const auto new_eq_map = this->template update_eq_map<Type>();
    static constexpr size_t FullDom =
        linalg::mat_cols_v<decltype(new_map.linear_part())>;

    linalg::Vector<T, FullDom> sel{};
    linalg::Vector<T, 1> bias{lift_map.translation()(0)};

    AK_ASSERT(idx < 0 && static_cast<size_t>(-idx) <= FullDom &&
              "Aux variable selector index out of bounds in "
              "CanonicalFormBuilder::make");

    const size_t selector_idx = FullDom - static_cast<size_t>(-idx);
    sel(selector_idx) = lift_map.linear_part()(0)(0);
    const linalg::RowVector<T, FullDom> sel_row = linalg::as_row_vector(sel);
    const auto new_coord_map =
        affine::AffineMap<decltype(sel_row)>(sel_row, bias);
    const auto new_cb =
        conic::ConeBlock(new_map, std::tuple_cat(cb.cones, cones));
    return CanonicalFormBuilder<DecVarDom, decltype(new_coord_map),
                                decltype(new_cb), decltype(new_eq_map)>(
        new_coord_map, new_cb, new_eq_map);
  }

  /**
   * @brief Append child canonical data without changing the selector.
   *
   * @param cones emitted cones from the child atom
   *
   * @return builder with updated cone block and equality map
   */
  template <convex::AtomType Type, typename... Cs>
  constexpr auto make(const std::tuple<Cs...>& cones) const {
    const auto new_map = this->template update_map<Type>();
    const auto new_eq_map = this->template update_eq_map<Type>();
    const auto new_cb =
        conic::ConeBlock(new_map, std::tuple_cat(cb.cones, cones));
    return CanonicalFormBuilder<DecVarDom, decltype(new_map), decltype(new_cb),
                                decltype(new_eq_map)>(new_map, new_cb,
                                                      new_eq_map);
  }

  CoordMap coord_map; ///< Affine map from decision/auxiliary columns to current
                      ///< atom coordinates.
  CB cb; ///< Accumulated cone block containing cone tuple and affine rows.
  EqMap eq_map; ///< Accumulated equality map in the same column layout.

private:
  /**
   * @brief Update the accumulated cone map for the next canonicalization step.
   *
   * Depending on child and operation update the cone block map F and g and
   * return resultant map
   *
   * @return affine map with preserved rows and any newly appended coordinate
   * rows
   */
  template <convex::AtomType Type>
  constexpr auto update_map() const {
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

    static constexpr bool is_op =
        Type == convex::AtomType::ADD || Type == convex::AtomType::CAP;
    static constexpr bool is_add = Type == convex::AtomType::ADD;

    /// (dakre) F cols: [ dec vars | cone aux | coord aux ]
    static constexpr size_t FullDom =
        is_add
            ? (N + ConeAux + CoordAux)
            : (is_op ? ((ConeDom > CoordDom) ? ConeDom : CoordDom) : CoordDom);
    static constexpr size_t FullCodom = ConeCodom + CoordCodom;

    static constexpr size_t CoordAuxStartCol =
        is_add ? ((ConeDom == 0) ? N : ConeDom) : N;
    static constexpr size_t CoordAuxEndCol =
        is_add ? (CoordAuxStartCol + CoordAux) : CoordDom;

    linalg::Matrix<T, FullCodom, FullDom> F{};
    linalg::Vector<T, FullCodom> g{};

    /// (0) fill upper left with prev cone map [ dec var | cone aux ]
    if constexpr (ConeDom != 0) {
      F = linalg::set_block<0, ConeCodom, 0, ConeDom>(F,
                                                      cone_map.linear_part());
      for (size_t i = 0; i < ConeCodom; ++i) g(i) = cone_map.translation()(i);
    }

    /// (1) fill bottom left from coord map
    const auto A = coord_map.linear_part();
    const auto dec_spanning_blk = linalg::get_block<0, CoordCodom, 0, N>(A);
    F = linalg::set_block<ConeCodom, FullCodom, 0, N>(F, dec_spanning_blk);

    /// (2) fill bottom right with new aux vars from latest cone
    if constexpr (Type != convex::AtomType::VAR &&
                  ((is_add && CoordAux > 0) || (!is_add && CoordDom > N))) {
      static_assert(CoordAuxEndCol > CoordAuxStartCol);
      const auto coord_aux_blk =
          linalg::get_block<0, CoordCodom, N, CoordDom>(A);
      F = linalg::set_block<ConeCodom, FullCodom, CoordAuxStartCol,
                            CoordAuxEndCol>(F, coord_aux_blk);
    }

    for (size_t i = 0; i < CoordCodom; ++i)
      g(i + ConeCodom) = coord_map.translation()(i);

    const auto new_map = affine::AffineMap<decltype(F)>(F, g);
    return new_map;
  }

  /**
   * @brief Update the accumulated equality map for the next step.
   *
   * Tracks the same column layout as the cone map so equality rows continue to
   * reference decision and auxiliary variables consistently.
   *
   * @return equality map with column layout aligned to the cone map
   */
  template <convex::AtomType Type>
  constexpr auto update_eq_map() const {
    using EqMapT = affine::affine_mat_t<EqMap>;

    static_assert(std::same_as<T, linalg::mat_type_t<EqMapT>>,
                  "type mismatch between equality map and coord map");

    static constexpr size_t CoordDom = linalg::mat_cols_v<CoordMapT>;
    static constexpr size_t CoordAux = CoordDom - N;

    static constexpr size_t EqDom = linalg::mat_cols_v<EqMapT>;
    static constexpr size_t EqCodom = linalg::mat_rows_v<EqMapT>;
    static constexpr size_t EqAux = EqDom == 0 ? 0 : EqDom - N;

    static constexpr bool is_op =
        Type == convex::AtomType::ADD || Type == convex::AtomType::CAP;
    static constexpr bool is_add = Type == convex::AtomType::ADD;

    static constexpr size_t FullDom =
        is_add ? (N + EqAux + CoordAux)
               : (is_op ? ((EqDom > CoordDom) ? EqDom : CoordDom) : CoordDom);

    static constexpr size_t CoordAuxStartCol =
        is_add ? ((EqDom == 0) ? N : EqDom) : N;
    static constexpr size_t CoordAuxEndCol =
        is_add ? (CoordAuxStartCol + CoordAux) : CoordDom;

    linalg::Matrix<T, EqCodom, FullDom> A{};
    linalg::Vector<T, EqCodom> b{};

    if constexpr (EqCodom != 0 && EqDom != 0) {
      A = linalg::set_block<0, EqCodom, 0, EqDom>(A, eq_map.linear_part());
      for (size_t i = 0; i < EqCodom; ++i) b(i) = eq_map.translation()(i);
    }

    if constexpr (EqCodom != 0 && Type != convex::AtomType::VAR &&
                  ((is_add && EqAux > 0) || (!is_add && EqDom > N))) {
      static_assert(CoordAuxEndCol > CoordAuxStartCol);
      const auto coord_aux_blk =
          linalg::get_block<0, EqCodom, N, CoordDom>(eq_map.linear_part());
      A = linalg::set_block<0, EqCodom, CoordAuxStartCol, CoordAuxEndCol>(
          A, coord_aux_blk);
    }

    return affine::AffineMap<decltype(A)>(A, b);
  }
};

/** @brief Create an empty cone block.
 *
 * @tparam T scalar field for the empty affine map
 *
 * @return empty cone block
 */
template <linalg::FieldLike T>
constexpr auto create_empty_cblk() {
  constexpr auto empty = std::make_tuple();
  constexpr conic::ConeBlock cb(
      affine::AffineMap(linalg::Matrix<T, 0, 0>{}, linalg::Vector<T, 0>{}),
      empty);
  return cb;
}

/** @brief Create an empty equality affine map.
 *
 * @tparam T scalar field for the empty affine map
 *
 * @return equality affine map
 */
template <linalg::FieldLike T>
constexpr auto create_empty_eq_map() {
  return affine::AffineMap(linalg::Matrix<T, 0, 0>{}, linalg::Vector<T, 0>{});
}

/** @brief Create an empty canonical-form builder.
 *
 * @tparam DecVarDom original decision-variable domain
 *
 * @return cone block builder
 */
template <linalg::RnSpace DecVarDom>
constexpr auto create_empty_cblk_builder() {
  using T = linalg::space_type_t<DecVarDom>;
  const auto cblk = create_empty_cblk<T>();
  const auto eq_map = create_empty_eq_map<T>();
  return CanonicalFormBuilder<DecVarDom, decltype(cblk.map), decltype(cblk),
                              decltype(eq_map)>(cblk.map, cblk, eq_map);
}

} // namespace conic
