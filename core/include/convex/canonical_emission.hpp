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
#include "convex/canonical_trace.hpp"
#include "convex/cone_traits.hpp"

namespace conic {

/**
 * @brief Empty canonical provenance trace.
 *
 * This is the zero-cost release/default trace carrier. Rich canonical
 * provenance can later specialize the same typed trace slot without changing
 * emission assembly APIs.
 */
using EmptyTrace = convex::EmptyCanonicalTrace;

/**
 * @brief Atom-local canonical emission bundle.
 *
 * Atoms own the local canonical recipe. This bundle names the result of that
 * recipe without implying runtime mutation or heap-backed growth: all fields
 * are fixed-size, constexpr-friendly types. The local coordinate map is not the
 * builder's global coordinate selector; it is the cone/equality recipe the
 * builder realizes into the global canonical layout.
 */
template <affine::AffineMapLike CoordMap, typename ConeTuple,
          affine::AffineMapLike EqMap, typename Trace = EmptyTrace>
struct LocalEmission {
  using Tuple = std::remove_cvref_t<ConeTuple>;

  static constexpr size_t local_coord_rows =
      linalg::mat_rows_v<affine::affine_mat_t<CoordMap>>;
  static constexpr size_t local_coord_cols =
      linalg::mat_cols_v<affine::affine_mat_t<CoordMap>>;
  static constexpr size_t cone_count = cone_tuple_num_cones_v<Tuple>;
  static constexpr size_t local_cone_dim = cone_tuple_local_cone_dim_v<Tuple>;
  static constexpr size_t local_eq_rows =
      linalg::mat_rows_v<affine::affine_mat_t<EqMap>>;
  static constexpr size_t local_eq_cols =
      linalg::mat_cols_v<affine::affine_mat_t<EqMap>>;

  /**
   * @brief CTOR
   *
   * @param coord_map_in (CoordMap)
   * @param cones_in (ConeTuple)
   * @param eq_map_in (EqMap)
   */
  constexpr explicit LocalEmission(const CoordMap& coord_map_in,
                                   const Tuple& cones_in,
                                   const EqMap& eq_map_in)
      : local_coord_map(coord_map_in), local_cones(cones_in),
        local_eq_map(eq_map_in) {}

  /** @brief public member vars */
  CoordMap local_coord_map; ///< Local coord map for emission
  ConeTuple local_cones;    ///< Local cones in emission
  EqMap local_eq_map;       ///< Local equality map
  Trace trace;              ///< Trace object
};

/**
 * @brief Atom-local equality emission bundle.
 *
 * Equality emissions are a separate canonical branch from cone emissions. They
 * carry affine rows that must hold with equality and do not model those rows as
 * nonnegative cone constraints.
 */
template <affine::AffineMapLike EqMap, typename Trace = EmptyTrace>
struct LocalEqualityEmission {
  static constexpr size_t local_eq_rows =
      linalg::mat_rows_v<affine::affine_mat_t<EqMap>>;
  static constexpr size_t local_eq_cols =
      linalg::mat_cols_v<affine::affine_mat_t<EqMap>>;

  constexpr explicit LocalEqualityEmission(const EqMap& eq_map_in)
      : local_eq_map(eq_map_in) {}

  /** @brief public member vars */
  EqMap local_eq_map; ///< Local equality map
  Trace trace;        ///< Trace object
};

/// @brief Deduce LocalEmission template arguments from fixed local components.
///
/// This keeps atom code readable as `LocalEmission(map, cones, eq_map)` while
/// preserving the exact constexpr component types. The omitted trace argument
/// uses the default EmptyTrace carrier.
template <typename CoordMap, typename ConeTuple, typename EqMap>
LocalEmission(const CoordMap&, const ConeTuple&, const EqMap&)
    -> LocalEmission<CoordMap, ConeTuple, EqMap>;

/// @brief Deduce LocalEmission template arguments when a trace is supplied.
template <typename CoordMap, typename ConeTuple, typename EqMap, typename Trace>
LocalEmission(const CoordMap&, const ConeTuple&, const EqMap&, const Trace&)
    -> LocalEmission<CoordMap, ConeTuple, EqMap, Trace>;

/// @brief Deduce LocalEqualityEmission template arguments from equality rows.
template <typename EqMap>
LocalEqualityEmission(const EqMap&) -> LocalEqualityEmission<EqMap>;

/// @brief Deduce LocalEqualityEmission template arguments when trace exists.
template <typename EqMap, typename Trace>
LocalEqualityEmission(const EqMap&, const Trace&)
    -> LocalEqualityEmission<EqMap, Trace>;

/// @brief Base traits for atom-local canonical emissions.
template <typename Emission>
struct emission_traits;

/// @brief Traits specialization for LocalEmission.
template <affine::AffineMapLike CoordMap, typename ConeTuple,
          affine::AffineMapLike EqMap, typename Trace>
struct emission_traits<LocalEmission<CoordMap, ConeTuple, EqMap, Trace>> {
  using LocalCoordinateMap = CoordMap;
  using Cones = ConeTuple;
  using LocalEqualityMap = EqMap;
  using TraceT = Trace;

  static constexpr size_t LocalCoordRows =
      LocalEmission<CoordMap, ConeTuple, EqMap, Trace>::local_coord_rows;
  static constexpr size_t LocalCoordCols =
      LocalEmission<CoordMap, ConeTuple, EqMap, Trace>::local_coord_cols;
  static constexpr size_t NumCones =
      LocalEmission<CoordMap, ConeTuple, EqMap, Trace>::cone_count;
  static constexpr size_t LocalConeDim =
      LocalEmission<CoordMap, ConeTuple, EqMap, Trace>::local_cone_dim;
  static constexpr size_t LocalEqRows =
      LocalEmission<CoordMap, ConeTuple, EqMap, Trace>::local_eq_rows;
  static constexpr size_t LocalEqCols =
      LocalEmission<CoordMap, ConeTuple, EqMap, Trace>::local_eq_cols;
};

/// @brief Traits specialization for LocalEqualityEmission.
template <affine::AffineMapLike EqMap, typename Trace>
struct emission_traits<LocalEqualityEmission<EqMap, Trace>> {
  using LocalEqualityMap = EqMap;
  using TraceT = Trace;

  static constexpr size_t LocalEqRows =
      LocalEqualityEmission<EqMap, Trace>::local_eq_rows;
  static constexpr size_t LocalEqCols =
      LocalEqualityEmission<EqMap, Trace>::local_eq_cols;
};

/// @brief Concept for fixed-size atom-local canonical emission bundles.
template <typename Emission>
concept LocalEmissionLike = requires(const Emission& emission) {
  { emission.local_coord_map };
  { emission.local_cones };
  { emission.local_eq_map };
  { emission.trace };
};

/// @brief Concept for fixed-size atom-local equality emission bundles.
template <typename Emission>
concept LocalEqualityEmissionLike =
    !LocalEmissionLike<Emission> && requires(const Emission& emission) {
      { emission.local_eq_map };
      { emission.trace };
    };

template <typename Emission>
using emission_local_coord_map_t = typename emission_traits<
    linalg::remove_cvref_t<Emission>>::LocalCoordinateMap;

template <typename Emission>
using emission_cones_t =
    typename emission_traits<linalg::remove_cvref_t<Emission>>::Cones;

template <typename Emission>
using emission_local_eq_map_t = typename emission_traits<
    linalg::remove_cvref_t<Emission>>::LocalEqualityMap;

template <typename Emission>
using emission_trace_t =
    typename emission_traits<linalg::remove_cvref_t<Emission>>::TraceT;

template <typename Emission>
inline constexpr size_t emission_local_coord_rows_v =
    emission_traits<linalg::remove_cvref_t<Emission>>::LocalCoordRows;

template <typename Emission>
inline constexpr size_t emission_local_coord_cols_v =
    emission_traits<linalg::remove_cvref_t<Emission>>::LocalCoordCols;

template <typename Emission>
inline constexpr size_t emission_num_cones_v =
    emission_traits<linalg::remove_cvref_t<Emission>>::NumCones;

template <typename Emission>
inline constexpr size_t emission_local_cone_dim_v =
    emission_traits<linalg::remove_cvref_t<Emission>>::LocalConeDim;

template <typename Emission>
inline constexpr size_t emission_local_eq_rows_v =
    emission_traits<linalg::remove_cvref_t<Emission>>::LocalEqRows;

template <typename Emission>
inline constexpr size_t emission_local_eq_cols_v =
    emission_traits<linalg::remove_cvref_t<Emission>>::LocalEqCols;

template <typename Emission>
using emission_coord_map_t = emission_local_coord_map_t<Emission>;

template <typename Emission>
using emission_eq_map_t = emission_local_eq_map_t<Emission>;

template <typename Emission>
inline constexpr size_t emission_coord_rows_v =
    emission_local_coord_rows_v<Emission>;

template <typename Emission>
inline constexpr size_t emission_coord_cols_v =
    emission_local_coord_cols_v<Emission>;

template <typename Emission>
inline constexpr size_t emission_cone_dim_v =
    emission_local_cone_dim_v<Emission>;

template <typename Emission>
inline constexpr size_t emission_eq_rows_v = emission_local_eq_rows_v<Emission>;

template <typename Emission>
inline constexpr size_t emission_eq_cols_v = emission_local_eq_cols_v<Emission>;

} // namespace conic
