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

#include "linalg/affine/affine_map.hpp"
#include "config/assert.hpp"
#include "convex/canonical_emission.hpp"
#include "convex/cone_traits.hpp"
#include "convex/convex_traits.hpp"

namespace conic {

/**
 * @brief Builder for conic canonical forms.
 *
 * The builder carries the current global coordinate selector, realized cone
 * block, and equality map while atom-local recipes expand the canonical type
 * layout during canonicalization.
 */
template <linalg::RnSpace DecVarDom, affine::AffineMapLike CoordMap,
          conic::ConeBlockLike CB, affine::AffineMapLike EqMap,
          typename CanonicalTrace = convex::EmptyCanonicalTrace>
struct CanonicalFormBuilder {
  /// @brief Affine matrix type for the current cone-coordinate selector.
  using CoordMapT = affine::affine_mat_t<CoordMap>;
  /// @brief Scalar field used by the canonical form.
  using T = linalg::matrix_scalar_t<CoordMapT>;

  /// @brief Number of original decision variables before auxiliary columns.
  static constexpr std::size_t N = linalg::space_dim_v<DecVarDom>;

  /**
   * @brief Construct a builder from existing canonical-form components.
   *
   * @param coord_in affine selector for the current atom coordinates
   * @param cb_in accumulated cone block
   * @param eq_in accumulated equality map
   */
  constexpr explicit CanonicalFormBuilder(const CoordMap& coord_in,
                                          const CB& cb_in, const EqMap& eq_in,
                                          const CanonicalTrace& trace_in = {})
      : coord_map(coord_in), cb(cb_in), eq_map(eq_in),
        canonical_trace(trace_in) {}

  /**
   * @brief make new builder with tuple of cones and input cone lift map
   * selecting upon input index
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
    static constexpr std::size_t FullDom =
        linalg::mat_cols_v<decltype(new_map.linear_part())>;

    linalg::Vector<T, FullDom> sel{};
    linalg::Vector<T, 1> bias{lift_map.translation()(0)};

    AK_ASSERT(idx < 0 && static_cast<std::size_t>(-idx) <= FullDom &&
              "Aux variable selector index out of bounds in "
              "CanonicalFormBuilder::make");

    const std::size_t selector_idx = FullDom - static_cast<std::size_t>(-idx);
    sel(selector_idx) = lift_map.linear_part()(0, 0);
    const linalg::RowVector<T, FullDom> sel_row = linalg::as_row_vector(sel);
    const auto new_coord_map =
        affine::AffineMap<decltype(sel_row)>(sel_row, bias);
    const auto new_cb =
        conic::ConeBlock(new_map, std::tuple_cat(cb.cones, cones));
    return make_builder<DecVarDom>(new_coord_map, new_cb, new_eq_map,
                                   canonical_trace);
  }

  /**
   * @brief make new builder based on tuple of cones (no input map or selector)
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
    return make_builder<DecVarDom>(new_map, new_cb, new_eq_map,
                                   canonical_trace);
  }

  /**
   * @brief Realize an atom-local emission and select a lifted coordinate based
   * on input cone lift map.
   *
   * @param idx negative index from the end of the realized coordinate domain
   * @param lift_map affine map that selects the lifted epigraph/hypograph
   * coordinate
   * @param emission fixed-size atom-local canonical recipe
   *
   * @return builder with the emission realized into the global canonical layout
   */
  template <convex::AtomType Type, affine::AffineMapLike LiftedMap,
            LocalEmissionLike Emission>
  constexpr auto realize(const int& idx, const LiftedMap& lift_map,
                         const Emission& emission) const {
    const auto traced = add_cone_trace<Type, Emission>(emission);
    const auto local_builder = make_builder<DecVarDom>(
        emission.local_coord_map, cb, emission.local_eq_map, traced);
    return local_builder.template make<Type>(idx, lift_map,
                                             emission.local_cones);
  }

  /**
   * @brief Realize an atom-local emission without changing the selector (no map
   * provided).
   *
   * @param emission fixed-size atom-local canonical recipe
   *
   * @return builder with the emission realized into the global canonical layout
   */
  template <convex::AtomType Type, LocalEmissionLike Emission>
  constexpr auto realize(const Emission& emission) const {
    const auto traced = add_cone_trace<Type, Emission>(emission);
    const auto local_builder = make_builder<DecVarDom>(
        emission.local_coord_map, cb, emission.local_eq_map, traced);
    return local_builder.template make<Type>(emission.local_cones);
  }

  /**
   * @brief Realize atom-local equality rows.
   *
   * This is the equality-only sister path to cone emission realization. It
   * merges affine equality rows into the builder equality map and leaves the
   * cone block and current coordinate selector unchanged.
   *
   * @param emission fixed-size atom-local equality recipe
   *
   * @return builder with the equality emission realized
   */
  template <LocalEqualityEmissionLike Emission>
  constexpr auto realize(const Emission& emission) const {
    using CurrentEqMat = affine::affine_mat_t<EqMap>;
    using LocalEqMap = emission_local_eq_map_t<Emission>;
    using LocalEqMat = affine::affine_mat_t<LocalEqMap>;

    static_assert(std::same_as<T, linalg::matrix_scalar_t<LocalEqMat>>,
                  "type mismatch between equality emission and builder");

    static constexpr std::size_t CurrentRows = linalg::mat_rows_v<CurrentEqMat>;
    static constexpr std::size_t CurrentCols = linalg::mat_cols_v<CurrentEqMat>;
    static constexpr std::size_t LocalRows = linalg::mat_rows_v<LocalEqMat>;
    static constexpr std::size_t LocalCols = linalg::mat_cols_v<LocalEqMat>;
    static constexpr std::size_t FullRows = CurrentRows + LocalRows;
    static constexpr std::size_t FullCols =
        CurrentCols > LocalCols ? CurrentCols : LocalCols;

    linalg::Matrix<T, FullRows, FullCols> A{};
    linalg::Vector<T, FullRows> b{};

    if constexpr (CurrentRows != 0 && CurrentCols != 0) {
      A = linalg::set_block<0, CurrentRows, 0, CurrentCols>(
          A, eq_map.linear_part());
      for (std::size_t i = 0; i < CurrentRows; ++i)
        b(i) = eq_map.translation()(i);
    }

    if constexpr (LocalRows != 0 && LocalCols != 0) {
      A = linalg::set_block<CurrentRows, FullRows, 0, LocalCols>(
          A, emission.local_eq_map.linear_part());
      for (std::size_t i = 0; i < LocalRows; ++i)
        b(i + CurrentRows) = emission.local_eq_map.translation()(i);
    }

    const auto new_eq_map = affine::AffineMap<decltype(A)>(A, b);
    const auto traced = add_equality_trace(emission);
    return make_builder<DecVarDom>(coord_map, cb, new_eq_map, traced);
  }

  /** @brief public member vars */
  CoordMap coord_map; ///< Global selector from canonical columns to current
                      ///< atom coordinates.
  CB cb;              ///< Realized cone block containing cone tuple and affine
                      ///< rows.
  EqMap eq_map;       ///< Realized equality map in the same column layout.
  CanonicalTrace canonical_trace; ///< Realized structural canonical trace.

private:
  /**
   * @brief Add cone trace from emission object
   *
   * @param emission (Emission- unused)
   *
   * @return canonical trace object
   */
  template <convex::AtomType Type, LocalEmissionLike Emission>
  constexpr auto add_cone_trace(const Emission& /*emission*/) const {
    using ConeTuple = emission_cones_t<Emission>;
    using EqMapT = affine::affine_mat_t<EqMap>;

    // The builder owns global offsets: atom emissions only describe local
    // cones/equalities, while the builder knows how many rows were realized
    // before this emission.
    constexpr std::size_t NodeId =
        convex::canonical_trace_num_nodes_v<CanonicalTrace>;
    constexpr std::size_t GlobalConeStart = conic::cone_block_num_cones_v<CB>;
    constexpr std::size_t ConeOffsetStart = conic::cone_block_map_rows_v<CB>;
    constexpr std::size_t LocalEqRows = emission_local_eq_rows_v<Emission>;
    constexpr std::size_t EqOffsetStart = linalg::mat_rows_v<EqMapT>;

    const auto nodes = std::make_tuple(convex::CanonicalNodeRecord{
        NodeId, convex::canonical_no_parent, Type, 0});
    const auto cones = convex::make_canonical_cone_records<ConeTuple>(
        NodeId, GlobalConeStart, ConeOffsetStart);

    if constexpr (LocalEqRows == 0) {
      return convex::extend_canonical_trace(canonical_trace, nodes, cones,
                                            std::make_tuple());
    } else {
      const auto equalities = std::make_tuple(convex::CanonicalEqualityRecord{
          NodeId, 0, EqOffsetStart, LocalEqRows});
      return convex::extend_canonical_trace(canonical_trace, nodes, cones,
                                            equalities);
    }
  }

  /**
   * @brief Add equality trace from emission object
   *
   * @param emission (Emission- unused)
   *
   * @return canonical trace object
   */
  template <LocalEqualityEmissionLike Emission>
  constexpr auto add_equality_trace(const Emission& /*emission*/) const {
    using EqMapT = affine::affine_mat_t<EqMap>;

    constexpr std::size_t NodeId =
        convex::canonical_trace_num_nodes_v<CanonicalTrace>;
    constexpr std::size_t LocalEqRows = emission_local_eq_rows_v<Emission>;
    constexpr std::size_t EqOffsetStart = linalg::mat_rows_v<EqMapT>;

    const auto nodes = std::make_tuple(convex::CanonicalNodeRecord{
        NodeId, convex::canonical_no_parent, convex::AtomType::SET, 0});

    if constexpr (LocalEqRows == 0) {
      return convex::extend_canonical_trace(
          canonical_trace, nodes, std::make_tuple(), std::make_tuple());
    } else {
      const auto equalities = std::make_tuple(convex::CanonicalEqualityRecord{
          NodeId, 0, EqOffsetStart, LocalEqRows});
      return convex::extend_canonical_trace(canonical_trace, nodes,
                                            std::make_tuple(), equalities);
    }
  }

  /**
   * @brief Update the accumulated cone map for the next canonicalization step.
   *
   * Depending on child and operation update the cone block map F and g and
   * return resultant map
   *
   * @return affine map with preserved rows and any newly realized coordinate
   * rows
   */
  template <convex::AtomType Type>
  constexpr auto update_map() const {
    const auto& cone_map = cb.map;
    using ConeMapT = affine::affine_mat_t<decltype(cone_map)>;

    static_assert(std::same_as<T, linalg::matrix_scalar_t<ConeMapT>>,
                  "type mismatch between cone map and coord map");

    static constexpr std::size_t CoordDom = linalg::mat_cols_v<CoordMapT>;
    static constexpr std::size_t CoordCodom = linalg::mat_rows_v<CoordMapT>;

    static constexpr std::size_t ConeDom = linalg::mat_cols_v<ConeMapT>;
    static constexpr std::size_t ConeCodom = linalg::mat_rows_v<ConeMapT>;
    static constexpr std::size_t ConeAux = ConeDom == 0 ? 0 : ConeDom - N;
    static constexpr std::size_t CoordAux = CoordDom - N;

    static constexpr bool is_op =
        Type == convex::AtomType::ADD || Type == convex::AtomType::CAP;
    static constexpr bool is_add = Type == convex::AtomType::ADD;

    /// (dakre) F cols: [ dec vars | cone aux | coord aux ]
    static constexpr std::size_t FullDom =
        is_add
            ? (N + ConeAux + CoordAux)
            : (is_op ? ((ConeDom > CoordDom) ? ConeDom : CoordDom) : CoordDom);
    static constexpr std::size_t FullCodom = ConeCodom + CoordCodom;

    static constexpr std::size_t CoordAuxStartCol =
        is_add ? ((ConeDom == 0) ? N : ConeDom) : N;
    static constexpr std::size_t CoordAuxEndCol =
        is_add ? (CoordAuxStartCol + CoordAux) : CoordDom;

    linalg::Matrix<T, FullCodom, FullDom> F{};
    linalg::Vector<T, FullCodom> g{};

    /// (0) fill upper left with prev cone map [ dec var | cone aux ]
    if constexpr (ConeDom != 0) {
      F = linalg::set_block<0, ConeCodom, 0, ConeDom>(F,
                                                      cone_map.linear_part());
      for (std::size_t i = 0; i < ConeCodom; ++i)
        g(i) = cone_map.translation()(i);
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

    for (std::size_t i = 0; i < CoordCodom; ++i)
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

    static_assert(std::same_as<T, linalg::matrix_scalar_t<EqMapT>>,
                  "type mismatch between equality map and coord map");

    static constexpr std::size_t CoordDom = linalg::mat_cols_v<CoordMapT>;
    static constexpr std::size_t CoordAux = CoordDom - N;

    static constexpr std::size_t EqDom = linalg::mat_cols_v<EqMapT>;
    static constexpr std::size_t EqCodom = linalg::mat_rows_v<EqMapT>;
    static constexpr std::size_t EqAux = EqDom == 0 ? 0 : EqDom - N;

    static constexpr bool is_op =
        Type == convex::AtomType::ADD || Type == convex::AtomType::CAP;
    static constexpr bool is_add = Type == convex::AtomType::ADD;

    static constexpr std::size_t FullDom =
        is_add ? (N + EqAux + CoordAux)
               : (is_op ? ((EqDom > CoordDom) ? EqDom : CoordDom) : CoordDom);

    static constexpr std::size_t CoordAuxStartCol =
        is_add ? ((EqDom == 0) ? N : EqDom) : N;
    static constexpr std::size_t CoordAuxEndCol =
        is_add ? (CoordAuxStartCol + CoordAux) : CoordDom;

    linalg::Matrix<T, EqCodom, FullDom> A{};
    linalg::Vector<T, EqCodom> b{};

    if constexpr (EqCodom != 0 && EqDom != 0) {
      A = linalg::set_block<0, EqCodom, 0, EqDom>(A, eq_map.linear_part());
      for (std::size_t i = 0; i < EqCodom; ++i) b(i) = eq_map.translation()(i);
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

/**
 * @brief Construct a canonical-form builder while deducing all realized
 * component types (aside from dec var dom)
 *
 * @param coord_map (CoordMap)
 * @param cb (ConeBlock)
 * @param eq_map (AffineMap)
 * @param canonical_trace (CanonicalTrace)
 *
 * @return builder
 */
template <linalg::RnSpace DecVarDom, affine::AffineMapLike CoordMap,
          conic::ConeBlockLike CB, affine::AffineMapLike EqMap,
          typename CanonicalTrace = convex::EmptyCanonicalTrace>
constexpr auto make_builder(const CoordMap& coord_map, const CB& cb,
                            const EqMap& eq_map,
                            const CanonicalTrace& canonical_trace = {}) {
  return CanonicalFormBuilder<DecVarDom, CoordMap, CB, EqMap, CanonicalTrace>(
      coord_map, cb, eq_map, canonical_trace);
}

/** @brief Create an empty cone block.
 *
 * @tparam T scalar field for the empty affine map
 *
 * @return empty cone block
 */
template <linalg::Field T>
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
template <linalg::Field T>
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
  return make_builder<DecVarDom>(cblk.map, cblk, eq_map);
}

} // namespace conic
