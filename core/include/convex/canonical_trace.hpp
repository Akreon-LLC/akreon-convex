/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "convex/cone_traits.hpp"
#include "convex/convex_traits.hpp"

namespace convex {

/// @brief Cone family emitted into the global cone block.
enum struct CanonicalConeKind : uint8_t {
  NONNEG_ORTHANT = 0,
  SECOND_ORDER = 1,
  POSITIVE_SEMIDEFINITE = 2,
  EXP = 3,
  POWER = 4,
  UNKNOWN = 255
};

/// @brief Sentinel used when a canonical node does not have a parent node.
inline constexpr std::size_t canonical_no_parent =
    std::numeric_limits<std::size_t>::max();

/**
 * @brief Structural atom emitted during canonicalization.
 *
 * A node represents one canonicalization emission site, usually a convex atom
 * or set operation. Cone and equality records reference `node_id` so downstream
 * layers can group solver slices by the atom/set that produced them.
 */
struct CanonicalNodeRecord {
  std::size_t node_id{}; ///< Stable index of this trace node.

  /// Parent trace node in a richer lineage DAG, or `canonical_no_parent` for
  /// root/flat emissions.
  std::size_t parent_id{canonical_no_parent};

  /// Canonicalization role that produced this node.
  AtomType atom_type{};

  /// Index of this node within its parent's children.
  std::size_t child_index{};
};

/**
 * @brief Cone slice emitted by one canonical trace node.
 *
 * `cone_offset` and `cone_dim` locate the slice inside the final cone primal,
 * dual, and residual vectors returned by the PD-IPM solver.
 */
struct CanonicalConeRecord {
  std::size_t node_id{};          ///< Node that emitted this cone.
  std::size_t local_cone_index{}; ///< Cone index within the emitting node.
  std::size_t
      global_cone_index{}; ///< Cone index in the accumulated cone block.
  CanonicalConeKind cone_kind{CanonicalConeKind::UNKNOWN};
  std::size_t cone_offset{}; ///< Starting row in the global cone vectors.
  std::size_t cone_dim{};    ///< Number of rows occupied by this cone.
};

/**
 * @brief Equality-row slice emitted by one canonical trace node.
 *
 * `row_offset` and `row_count` locate the slice inside the final equality dual
 * and equality residual vectors returned by the PD-IPM solver.
 */
struct CanonicalEqualityRecord {
  std::size_t node_id{}; ///< Node that emitted these equality rows.
  std::size_t
      local_eq_index{};     ///< Equality block index within the emitting node.
  std::size_t row_offset{}; ///< Starting row in the global equality vectors.
  std::size_t row_count{};  ///< Number of equality rows in this slice.
};

/**
 * @brief Compile-time canonicalization trace.
 *
 * The trace is tuple-backed to preserve constexpr/no-heap behavior while
 * carrying enough structural metadata for runtime bindings to map final
 * primal-dual values back to canonical atoms, cones, and equalities.
 */
template <typename NodeRecords = std::tuple<>,
          typename ConeRecords = std::tuple<>,
          typename EqualityRecords = std::tuple<>>
struct CanonicalTrace {
  NodeRecords nodes{};          ///< Canonical node records.
  ConeRecords cones{};          ///< Cone slice records.
  EqualityRecords equalities{}; ///< Equality slice records.
};

using EmptyCanonicalTrace = CanonicalTrace<>;

template <typename Trace>
struct canonical_trace_traits;

template <typename NodeTuple, typename ConeTuple, typename EqualityTuple>
struct canonical_trace_traits<
    CanonicalTrace<NodeTuple, ConeTuple, EqualityTuple>> {
  using Nodes = NodeTuple;
  using Cones = ConeTuple;
  using Equalities = EqualityTuple;

  static constexpr std::size_t NumNodes = std::tuple_size_v<NodeTuple>;
  static constexpr std::size_t NumCones = std::tuple_size_v<ConeTuple>;
  static constexpr std::size_t NumEqualities = std::tuple_size_v<EqualityTuple>;
};

template <typename Trace>
using canonical_trace_nodes_t =
    typename canonical_trace_traits<linalg::remove_cvref_t<Trace>>::Nodes;

template <typename Trace>
using canonical_trace_cones_t =
    typename canonical_trace_traits<linalg::remove_cvref_t<Trace>>::Cones;

template <typename Trace>
using canonical_trace_equalities_t =
    typename canonical_trace_traits<linalg::remove_cvref_t<Trace>>::Equalities;

template <typename Trace>
inline constexpr std::size_t canonical_trace_num_nodes_v =
    canonical_trace_traits<linalg::remove_cvref_t<Trace>>::NumNodes;

template <typename Trace>
inline constexpr std::size_t canonical_trace_num_cones_v =
    canonical_trace_traits<linalg::remove_cvref_t<Trace>>::NumCones;

template <typename Trace>
inline constexpr std::size_t canonical_trace_num_equalities_v =
    canonical_trace_traits<linalg::remove_cvref_t<Trace>>::NumEqualities;

template <typename Cone>
struct canonical_cone_kind {
  static constexpr CanonicalConeKind value = CanonicalConeKind::UNKNOWN;
};

template <linalg::VectorContainer Vec>
struct canonical_cone_kind<conic::NonnegOrthantCone<Vec>> {
  static constexpr CanonicalConeKind value = CanonicalConeKind::NONNEG_ORTHANT;
};

template <linalg::VectorContainer Vec>
struct canonical_cone_kind<conic::SecondOrderCone<Vec>> {
  static constexpr CanonicalConeKind value = CanonicalConeKind::SECOND_ORDER;
};

template <linalg::VectorContainer Vec, std::size_t Dim>
struct canonical_cone_kind<conic::PositiveSemidefiniteCone<Vec, Dim>> {
  static constexpr CanonicalConeKind value =
      CanonicalConeKind::POSITIVE_SEMIDEFINITE;
};

template <linalg::VectorContainer Vec>
struct canonical_cone_kind<conic::ExpCone<Vec>> {
  static constexpr CanonicalConeKind value = CanonicalConeKind::EXP;
};

template <linalg::VectorContainer Vec>
struct canonical_cone_kind<conic::PowerCone<Vec>> {
  static constexpr CanonicalConeKind value = CanonicalConeKind::POWER;
};

template <typename Cone>
inline constexpr CanonicalConeKind canonical_cone_kind_v =
    canonical_cone_kind<linalg::remove_cvref_t<Cone>>::value;

template <typename ConeTuple, std::size_t Index>
consteval std::size_t cone_tuple_offset_before_v() {
  using Tuple = linalg::remove_cvref_t<ConeTuple>;
  if constexpr (Index == 0) {
    return 0;
  } else {
    using PrevCone = std::tuple_element_t<Index - 1, Tuple>;
    return cone_tuple_offset_before_v<ConeTuple, Index - 1>() +
           conic::cone_dim_t<PrevCone>;
  }
}

/**
 * @brief Extends canonical trace by tuple concatenation.
 *
 * @param trace (Trace)
 * @param nodes (NodeRecords)
 * @param cones (ConeRecords)
 * @param equalities (EqualityRecords)
 *
 * @return canonical trace object
 */
template <typename Trace, typename NodeRecords, typename ConeRecords,
          typename EqualityRecords>
constexpr auto extend_canonical_trace(const Trace& trace,
                                      const NodeRecords& nodes,
                                      const ConeRecords& cones,
                                      const EqualityRecords& equalities) {
  return CanonicalTrace<decltype(std::tuple_cat(trace.nodes, nodes)),
                        decltype(std::tuple_cat(trace.cones, cones)),
                        decltype(std::tuple_cat(trace.equalities, equalities))>{
      std::tuple_cat(trace.nodes, nodes), std::tuple_cat(trace.cones, cones),
      std::tuple_cat(trace.equalities, equalities)};
}

/**
 * @brief Make canonical cone records for a node-emitted tuple of cones.
 *
 * @param node_id (std::size_t)
 * @param global_cone_state (std::size_t)
 * @param cone_offset_start (std::size_t)
 *
 * @return canonical trace object
 */
template <typename ConeTuple>
constexpr auto make_canonical_cone_records(std::size_t node_id,
                                           std::size_t global_cone_start,
                                           std::size_t cone_offset_start) {
  using Tuple = linalg::remove_cvref_t<ConeTuple>;

  auto make_records = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::make_tuple(CanonicalConeRecord{
        node_id, Is, global_cone_start + Is,
        canonical_cone_kind_v<std::tuple_element_t<Is, Tuple>>,
        cone_offset_start + cone_tuple_offset_before_v<Tuple, Is>(),
        conic::cone_dim_t<std::tuple_element_t<Is, Tuple>>}...);
  };

  return make_records(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

} // namespace convex
