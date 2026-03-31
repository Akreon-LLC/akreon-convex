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
#include "cone_traits.hpp"

namespace convex {

/// @brief alias for lifted var map
template <typename T>
using LiftedVarMat = linalg::Matrix<T, 1, 1>;

/// @brief alias for lifted var map
template <typename T>
using LiftedVarBias = linalg::Vector<T, 1>;

/// @brief Atom category used by canonicalization and DCP checks.
enum struct AtomType : uint8_t { VAR = 0, INDEX, FN, SET, ADD, CAP };

/// @brief Domain/variable annotation attached to domains/codomains.
enum struct VarType : uint8_t {
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

/// @brief Curvature annotation for atom codomain.
enum struct CurvatureType : uint8_t { AFFINE = 0, CONVEX, CONCAVE };

/// @brief Monotonicity annotation for function atoms.
enum struct MonotonicityType : uint8_t { UNKNOWN = 0, NONDEC, NONINC };

/// @brief InequalityType for inequality atom.
enum struct InequalityType : uint8_t { EQ = 0, LEQ, GEQ };

/// @brief Problem construction lifecycle stage.
enum struct StageType : uint8_t { INIT = 0, IN_PROGRESS, PROBLEM_SET };

/// @brief Domain wrapper with a static sign label.
template <linalg::RnSpace Space, VarType Sign>
struct ConvexDomain {};

/// @brief Base domain traits.
template <typename Dom>
struct dom_traits {};

template <linalg::RnSpace SpaceIn, VarType Sign>
struct dom_traits<ConvexDomain<SpaceIn, Sign>> {
  using Space = SpaceIn;
  using T = linalg::space_type_t<Space>;
  using Point = linalg::space_point_t<Space>;
  static constexpr size_t N = linalg::space_dim_v<Space>;
  static constexpr VarType sign = Sign;
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
inline constexpr VarType dom_vartype_v =
    dom_traits<linalg::remove_cvref_t<Dom>>::sign;

/// @brief Atom metadata bundling domain, codomain, and curvature.
template <typename Dom, typename Codom, CurvatureType Curv>
  requires(std::same_as<dom_type_t<Dom>, dom_type_t<Codom>>)
struct ConvexAtomMeta {};

/// @brief Base metadata traits.
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

/// @brief forward declaration - huber
template <typename Meta, typename Child>
struct Huber;

/// @brief forward declaration - sum op
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
template <typename Meta, typename Child, linalg::SquareMatrixLike MatIn>
struct LMI;

/// @brief simple enum to describe ball type
enum struct NormBallType : uint8_t { L2 = 0, L1, Linf };

/// @brief forward declaration of norm ball set
template <typename Meta, typename Child, NormBallType Type>
struct NormBall;

/// @brief forward declaration of polyhedron set
template <typename Meta, typename Child>
struct Polyhedron;

/// @brief forward declaration of set intersection operator.
template <typename Meta, typename... Cs>
struct Intersection;

/// @brief forward declaration of convex problem
template <conic::BuilderLike Builder, StageType Stage>
struct ConvexProblem;

/// @brief problem type traits
template <typename Prob>
struct prob_traits;

/// @brief problem type traits
template <conic::BuilderLike Builder, StageType Stage>
struct prob_traits<ConvexProblem<Builder, Stage>> {
  using BuilderT = Builder;
  static constexpr StageType StageV = Stage;
};

/// @brief builder type from prob
template <typename Prob>
using prob_builder_t =
    typename prob_traits<linalg::remove_cvref_t<Prob>>::BuilderT;

/// @brief stage type from prob
template <typename Prob>
inline constexpr bool is_problem_set_v =
    prob_traits<linalg::remove_cvref_t<Prob>>::StageV == StageType::PROBLEM_SET;

/// @brief ProblemLike Concept
template <typename Problem>
concept ProblemLike = requires(const Problem& prob) {
  { prob.linear_objective() };
  { prob.constraint_map() };
  { prob.constraint_cones() };
  { prob.equality_map() };
};

/// @brief Base atom traits.
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
  static constexpr AtomType Type = AtomType::INDEX;
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

template <typename Meta, typename Child>
struct atom_traits<Huber<Meta, Child>> {
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

template <typename Meta, typename Child, typename Mat>
struct atom_traits<LMI<Meta, Child, Mat>> {
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

/// @brief Domain alias for sum atom.
template <linalg::RnSpace Dom>
using SumDom = ConvexDomain<Dom, VarType::FREE>;

/// @brief Codomain alias for sum atom (scalar).
template <linalg::RnSpace Codom>
  requires(linalg::space_dim_v<Codom> == 1)
using SumCodom = ConvexDomain<Codom, VarType::FREE>;

/// @brief Meta alias for sum atom.
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using SumMeta =
    ConvexAtomMeta<SumDom<Dom>, SumCodom<Codom>, CurvatureType::AFFINE>;

/// @brief type traits for sum
template <linalg::RnSpace Dom, linalg::RnSpace Codom, typename... Cs>
struct atom_traits<Sum<SumMeta<Dom, Codom>, Cs...>> {
  using ConvexMeta = SumMeta<Dom, Codom>;
  static constexpr AtomType Type = AtomType::ADD;
};

/// @brief Domain alias for intersection atom.
template <linalg::RnSpace Space>
using IntersectionDom = ConvexDomain<Space, VarType::FREE>;

/// @brief Codomain alias for intersection atom.
template <linalg::RnSpace Space>
using IntersectionCodom = IntersectionDom<Space>;

/// @brief Meta alias for intersection atom.
template <linalg::RnSpace Space>
using IntersectionMeta =
    ConvexAtomMeta<IntersectionDom<Space>, IntersectionCodom<Space>,
                   CurvatureType::CONVEX>;

/// @brief type traits for intersection
template <linalg::RnSpace Space, typename... Cs>
struct atom_traits<Intersection<IntersectionMeta<Space>, Cs...>> {
  using ConvexMeta = IntersectionMeta<Space>;
  static constexpr AtomType Type = AtomType::CAP;
};

/// @brief Concept for any convex atom node in the expression tree.
template <typename Atom>
concept AtomLike = requires(const Atom& atom) {
  { atom.validate_domain() } -> std::convertible_to<void>;
  { atom.children };
};

/// @brief forward declaration of inequality
template <InequalityType IType, AtomLike LHS, typename RHS>
struct Inequality;

/// @brief normalize atom-facing types to a stable internal form.
template <typename T>
using atom_decay_t = linalg::remove_cvref_t<T>;

/// @brief Extract metadata type from atom.
template <AtomLike Atom>
using atom_meta_t =
    typename atom_traits<atom_decay_t<Atom>>::ConvexMeta;

/// @brief Compile-time atom category.
template <AtomLike Atom>
inline constexpr AtomType atom_type_v =
    atom_traits<atom_decay_t<Atom>>::Type;

/// @brief Helper predicates on atom category.
template <AtomLike Atom>
inline constexpr bool is_fn_type_v = atom_type_v<Atom> == AtomType::FN;

template <AtomLike Atom>
inline constexpr bool is_var_type_v = atom_type_v<Atom> == AtomType::VAR;

template <AtomLike Atom>
inline constexpr bool is_set_type_v = atom_type_v<Atom> == AtomType::SET;

template <AtomLike Atom>
inline constexpr bool is_add_type_v = atom_type_v<Atom> == AtomType::ADD;

template <AtomLike Atom>
inline constexpr bool is_cap_type_v = atom_type_v<Atom> == AtomType::CAP;

template <AtomLike Atom>
inline constexpr bool is_index_type_v = atom_type_v<Atom> == AtomType::INDEX;

}  // namespace convex
