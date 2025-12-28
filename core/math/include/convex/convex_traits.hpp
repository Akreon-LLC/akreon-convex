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

#include "affine/affine_traits.hpp"

namespace convex {

enum struct SignType : uint8_t {
  FREE = 0,
  NONNEG,
  NONPOS,
  POS,
  NEG,
  PSD,
  NSD,
  PD,
  ND
};

enum struct CurvatureType : uint8_t { AFFINE = 0, CONVEX, CONCAVE };

enum struct MonotonicityType : uint8_t { UNKNOWN = 0, NONDEC, NONINC };

enum struct ConstraintType : uint8_t { EQ = 0, LEQ, GEQ, IN_SET };

template <linalg::RnSpace Space, SignType Sign>
struct ConvexDomain {};

template <typename Dom>
struct dom_traits {};

template <linalg::RnSpace Space, SignType Sign>
struct dom_traits<ConvexDomain<Space, Sign>> {
  using T = linalg::space_type_t<Space>;
  using Point = linalg::space_point_t<Space>;
  static constexpr size_t N = linalg::space_dim_v<Space>;
  static constexpr SignType sign = Sign;
};

template <typename Dom>
using dom_type_t = typename dom_traits<linalg::remove_cvref_t<Dom>>::T;

template <typename Dom>
using dom_point_t = typename dom_traits<linalg::remove_cvref_t<Dom>>::Point;

template <typename Dom>
inline constexpr size_t dom_size_v = dom_traits<linalg::remove_cvref_t<Dom>>::N;

template <typename Dom>
inline constexpr SignType dom_sign_v =
    dom_traits<linalg::remove_cvref_t<Dom>>::sign;

template <typename Dom, typename Codom, CurvatureType Curv>
  requires(std::same_as<dom_type_t<Dom>, dom_type_t<Codom>>)
struct ConvexAtomMeta {};

template <typename Meta>
struct meta_traits {};

template <typename Dom, typename Codom, CurvatureType Curv>
struct meta_traits<ConvexAtomMeta<Dom, Codom, Curv>> {
  using Domain = Dom;
  using Codomain = Codom;
  static constexpr CurvatureType curvature = Curv;
};

template <typename Meta>
using meta_dom_t = typename meta_traits<linalg::remove_cvref_t<Meta>>::Domain;

template <typename Meta>
using meta_codom_t =
    typename meta_traits<linalg::remove_cvref_t<Meta>>::Codomain;

template <typename Meta>
inline constexpr CurvatureType meta_curv_v =
    meta_traits<linalg::remove_cvref_t<Meta>>::curvature;

template <typename Meta, typename Derived, typename... Cs>
struct Atom;

template <typename Meta>
struct Variable;

template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct ScalarView;

/// @brief forward declaration - convex fn
template <typename Meta, typename Derived, typename Child>
struct ConvexFn;

/// @brief forward declaration - max fn
template <typename Meta>
struct MaxFn;

/// @brief forward declaration - norm fn
template <typename Meta, linalg::NormLike Norm>
  requires(dom_size_v<dom_type_t<Meta>> == linalg::norm_size_v<Norm>)
struct NormFn;

/// @brief forward declaration - geo mean fn
template <typename Meta>
struct GeoMeanFn;

/// @brief forward declaration - exp fn
template <typename Meta>
  requires(dom_size_v<dom_type_t<Meta>> == 1)
struct ExpFn;

/// @brief forward declaration - log fn
template <typename Meta>
  requires(dom_size_v<dom_type_t<Meta>> == 1)
struct LogFn;

/// @brief forward declaration - square fn
template <typename Meta>
  requires(dom_size_v<dom_type_t<Meta>> == 1)
struct SquareFn;

/// @brief forward declaration - abs fn
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct AbsFn;

/// @brief forward declaration - inv pos fn
template <typename Meta>
  requires(dom_size_v<dom_type_t<Meta>> == 1)
struct InvPosFn;

/// @brief forward declaration - log sum exp fn
template <typename Meta>
struct LogSumExpFn;

// (dakre) TODO currently unimplemented but forward dec stub here

/// @brief forward declaration - log det fn
template <typename Meta>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<dom_type_t<Meta>>>)
struct LogDetFn;

/// @brief forward declaration - harmonic mean fn
template <typename Meta>
struct HarmonicMeanFn;

/// @brief forward declaration - sigmoid fn
template <typename Meta>
struct SigmoidFn;

/// @brief forward declaration - huber fn
template <typename Meta>
struct HuberFn;

/// @brief forward declaration - kl divergence
template <typename Meta>
struct KLDivergenceFn;

template <typename Atom>
struct atom_traits {};

template <typename Meta>
struct atom_traits<Variable<Meta>> : std::true_type {
  using ConvexMeta = Meta;
};

template <typename Meta, typename Child>
struct atom_traits<ScalarView<Meta, Child>> : std::true_type {
  using ConvexMeta = Meta;
};

template <typename Meta, typename Derived, typename Child>
struct atom_traits<ConvexFn<Meta, Derived, Child>> : std::false_type {
  using ConvexMeta = Meta;
};

template <typename Meta, typename Child>
struct atom_traits<Atom<Meta, Child>> : std::false_type {
  using ConvexMeta = Meta;
};

template <typename Atom>
using atom_meta_t =
    typename atom_traits<linalg::remove_cvref_t<Atom>>::ConvexMeta;

template <typename Atom>
inline constexpr bool is_var_type_v =
    atom_traits<linalg::remove_cvref_t<Atom>>{};

}  // namespace convex
