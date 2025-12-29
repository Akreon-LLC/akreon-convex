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

enum struct AtomType : uint8_t { VAR = 0, OP, FN };

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

template <linalg::RnSpace SpaceIn, SignType Sign>
struct dom_traits<ConvexDomain<SpaceIn, Sign>> {
  using Space = SpaceIn;
  using T = linalg::space_type_t<Space>;
  using Point = linalg::space_point_t<Space>;
  static constexpr size_t N = linalg::space_dim_v<Space>;
  static constexpr SignType sign = Sign;
};

template <typename Dom>
using dom_space_t = typename dom_traits<linalg::remove_cvref_t<Dom>>::Space;

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
  requires(dom_size_v<meta_dom_t<Meta>> == linalg::norm_size_v<Norm>)
struct NormFn;

/// @brief forward declaration - geo mean fn
template <typename Meta, typename Child>
struct GeoMean;

/// @brief forward declaration - exp fn
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct Exp;

/// @brief forward declaration - log fn
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct Log;

/// @brief forward declaration - square fn
template <typename Meta>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct SquareFn;

/// @brief forward declaration - abs fn
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct Abs;

/// @brief forward declaration - inv pos fn
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct InvPos;

/// @brief forward declaration - log sum exp fn
template <typename Meta>
struct LogSumExpFn;

template <typename Meta, typename... Cs>
struct Sum;

// (dakre) TODO currently unimplemented but forward dec stub here
#if 0
/// @brief forward declaration - log det fn
template <typename Meta>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<meta_type_t<Meta>>>)
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
#endif

template <typename Atom>
struct atom_traits {};

template <typename Meta, typename Child>
struct atom_traits<Atom<Meta, Child>> : std::false_type {
  using ConvexMeta = Meta;
};

template <typename Meta>
struct atom_traits<Variable<Meta>> : std::false_type {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::VAR;
};

template <typename Meta, typename Child>
struct atom_traits<ScalarView<Meta, Child>> : std::false_type {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::VAR;
};

template <typename Meta, typename Derived, typename Child>
struct atom_traits<ConvexFn<Meta, Derived, Child>> : std::true_type {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child>
struct atom_traits<Abs<Meta, Child>> : std::true_type {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child>
struct atom_traits<Exp<Meta, Child>> : std::true_type {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child>
struct atom_traits<GeoMean<Meta, Child>> : std::true_type {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child>
struct atom_traits<InvPos<Meta, Child>> : std::true_type {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child>
struct atom_traits<Log<Meta, Child>> : std::true_type {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <linalg::RnSpace Dom>
using SumDom = ConvexDomain<Dom, SignType::FREE>;

template <linalg::RnSpace Codom>
  requires(linalg::space_dim_v<Codom> == 1)
using SumCodom = ConvexDomain<Codom, SignType::FREE>;

template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using SumAffineMeta =
    ConvexAtomMeta<SumDom<Dom>, SumCodom<Codom>, CurvatureType::AFFINE>;

template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using SumConvexMeta =
    ConvexAtomMeta<SumDom<Dom>, SumCodom<Codom>, CurvatureType::CONVEX>;

template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using SumConcaveMeta =
    ConvexAtomMeta<SumDom<Dom>, SumCodom<Codom>, CurvatureType::CONCAVE>;

template <linalg::RnSpace Dom, linalg::RnSpace Codom, typename... Cs>
struct atom_traits<Sum<SumConvexMeta<Dom, Codom>, Cs...>> : std::false_type {
  using ConvexMeta = SumConvexMeta<Dom, Codom>;
  static constexpr AtomType Type = AtomType::OP;
};

template <linalg::RnSpace Dom, linalg::RnSpace Codom, typename... Cs>
struct atom_traits<Sum<SumConcaveMeta<Dom, Codom>, Cs...>> : std::false_type {
  using ConvexMeta = SumConcaveMeta<Dom, Codom>;
  static constexpr AtomType Type = AtomType::OP;
};

template <typename Atom>
using atom_meta_t =
    typename atom_traits<linalg::remove_cvref_t<Atom>>::ConvexMeta;

template <typename Atom>
inline constexpr bool is_fn_type_v =
    atom_traits<linalg::remove_cvref_t<Atom>>::Type == AtomType::FN;

template <typename Atom>
inline constexpr bool is_var_type_v =
    atom_traits<linalg::remove_cvref_t<Atom>>::Type == AtomType::VAR;

template <typename Atom>
inline constexpr bool is_op_type_v =
    atom_traits<linalg::remove_cvref_t<Atom>>::Type == AtomType::OP;

}  // namespace convex
