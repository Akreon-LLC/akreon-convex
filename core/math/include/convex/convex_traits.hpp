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

/// @brief alias for lifted var map
template <typename T>
using LiftedVarMat = linalg::Matrix<T, 1, 1>;

/// @brief alias for lifted var map
template <typename T>
using LiftedVarBias = linalg::Vector<T, 1>;

enum struct AtomType : uint8_t { VAR = 0, FN, SET, ADD, CAP };

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

/// @brief forward declaration - max fn
template <typename Meta, typename Child>
struct Max;

/// @brief forward declaration - norm fn
template <typename Meta, typename Child, linalg::NormLike NormFunc>
  requires(dom_size_v<meta_dom_t<Meta>> == linalg::norm_size_v<NormFunc>)
struct Norm;

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
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct Square;

/// @brief forward declaration - abs fn
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct Abs;

/// @brief forward declaration - inv pos fn
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct InvPos;

/// @brief forward declaration - log sum exp fn
template <typename Meta, typename Child>
struct LogSumExp;

/// @brief forward declaration - log sum exp fn
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct Sqrt;

/// @brief forward declaration - softplus fn
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct Softplus;

/// @brief forward declaration - harmonic mean fn
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 2)
struct HarmonicMean;

/// @brief forward declaration - kl divergence
template <typename Meta, typename Child>
struct KLDivergence;

// (dakre) TODO currently unimplemented but forward dec stub here
#if 0
/// @brief forward declaration - log det fn
template <typename Meta>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<meta_type_t<Meta>>>)
struct LogDetFn;

/// @brief forward declaration - huber fn
template <typename Meta>
struct HuberFn;
#endif

/// @brief forward declataion - sum op
template <typename Meta, typename... Cs>
struct Sum;

/// @brief forward declaration of convex set
template <typename Meta, typename Derived, typename Child>
struct ConvexSet;

/// @brief forward declaration of affine subspace
template <typename Meta, typename Child>
struct AffineSubspace;

/// @brief forward declaration of ellipsoid set
template <typename Meta, typename Child>
struct Ellipsoid;

/// @brief simple enum to describe type of halfspace
enum struct HalfspaceType : uint8_t { LH = 0, UH };

/// @brief forward declaration of halfspace
template <typename Meta, typename Child, HalfspaceType Type>
struct Halfspace;

/// @brief forward declaration of hyperplane
template <typename Meta, typename Child>
struct Hyperplane;

/// @brief forward declaration of LMI type
template <typename Meta, typename Child>
struct LMI;

/// @brief forward declaration of simplex set
template <typename Meta, typename Child>
struct Simplex;

/// @brief simple enum to describe ball type
enum struct NormBallType : uint8_t { L2 = 0, L1, Linf };

/// @brief forward declaration of norm ball set
template <typename Meta, typename Child, NormBallType Type>
struct NormBall;

/// @brief forward declaration of polyhedron set
template <typename Meta, typename Child>
struct Polyhedron;

template <typename Meta, typename... Cs>
struct Intersection;

template <typename Atom>
struct atom_traits {};

template <typename Meta, typename Child>
struct atom_traits<Atom<Meta, Child>> {
  using ConvexMeta = Meta;
};

template <typename Meta>
struct atom_traits<Variable<Meta>> : std::false_type {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::VAR;
};

template <typename Meta, typename Child>
struct atom_traits<ScalarView<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::VAR;
};

template <typename Meta, typename Child>
struct atom_traits<Abs<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child>
struct atom_traits<Exp<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child>
struct atom_traits<GeoMean<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child>
struct atom_traits<InvPos<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child>
struct atom_traits<Log<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child>
struct atom_traits<LogSumExp<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child>
struct atom_traits<Max<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child>
struct atom_traits<Square<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child, linalg::NormLike NormFunc>
struct atom_traits<Norm<Meta, Child, NormFunc>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child>
struct atom_traits<Sqrt<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child>
struct atom_traits<Softplus<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child>
struct atom_traits<HarmonicMean<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Child>
struct atom_traits<KLDivergence<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::FN;
};

template <typename Meta, typename Derived, typename Child>
struct atom_traits<ConvexSet<Meta, Derived, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::SET;
};

template <typename Meta, typename Child>
struct atom_traits<AffineSubspace<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::SET;
};

template <typename Meta, typename Child>
struct atom_traits<Ellipsoid<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::SET;
};

template <typename Meta, typename Child, HalfspaceType HType>
struct atom_traits<Halfspace<Meta, Child, HType>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::SET;
};

template <typename Meta, typename Child>
struct atom_traits<Hyperplane<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::SET;
};

template <typename Meta, typename Child>
struct atom_traits<LMI<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::SET;
};

template <typename Meta, typename Child>
struct atom_traits<Simplex<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::SET;
};

template <typename Meta, typename Child, NormBallType BType>
struct atom_traits<NormBall<Meta, Child, BType>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::SET;
};

template <typename Meta, typename Child>
struct atom_traits<Polyhedron<Meta, Child>> {
  using ConvexMeta = Meta;
  static constexpr AtomType Type = AtomType::SET;
};

template <linalg::RnSpace Dom>
using SumDom = ConvexDomain<Dom, SignType::FREE>;

template <linalg::RnSpace Codom>
  requires(linalg::space_dim_v<Codom> == 1)
using SumCodom = ConvexDomain<Codom, SignType::FREE>;

template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using SumConvexMeta =
    ConvexAtomMeta<SumDom<Dom>, SumCodom<Codom>, CurvatureType::CONVEX>;

template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using SumConcaveMeta =
    ConvexAtomMeta<SumDom<Dom>, SumCodom<Codom>, CurvatureType::CONCAVE>;

template <linalg::RnSpace Dom, linalg::RnSpace Codom, typename... Cs>
struct atom_traits<Sum<SumConvexMeta<Dom, Codom>, Cs...>> {
  using ConvexMeta = SumConvexMeta<Dom, Codom>;
  static constexpr AtomType Type = AtomType::ADD;
};

template <linalg::RnSpace Dom, linalg::RnSpace Codom, typename... Cs>
struct atom_traits<Sum<SumConcaveMeta<Dom, Codom>, Cs...>> {
  using ConvexMeta = SumConcaveMeta<Dom, Codom>;
  static constexpr AtomType Type = AtomType::ADD;
};

template <linalg::RnSpace Space>
using IntersectionDom = ConvexDomain<Space, SignType::FREE>;

template <linalg::RnSpace Space>
using IntersectionCodom = IntersectionDom<Space>;

template <linalg::RnSpace Space>
using IntersectionMeta =
    ConvexAtomMeta<IntersectionDom<Space>, IntersectionCodom<Space>,
                   CurvatureType::CONVEX>;

template <linalg::RnSpace Space, typename... Cs>
struct atom_traits<Intersection<IntersectionMeta<Space>, Cs...>> {
  using ConvexMeta = IntersectionMeta<Space>;
  static constexpr AtomType Type = AtomType::CAP;
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
inline constexpr bool is_set_type_v =
    atom_traits<linalg::remove_cvref_t<Atom>>::Type == AtomType::SET;

template <typename Atom>
inline constexpr bool is_add_type_v =
    atom_traits<linalg::remove_cvref_t<Atom>>::Type == AtomType::ADD;

template <typename Atom>
inline constexpr bool is_cap_type_v =
    atom_traits<linalg::remove_cvref_t<Atom>>::Type == AtomType::CAP;

}  // namespace convex
