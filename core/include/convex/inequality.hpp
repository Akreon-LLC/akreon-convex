/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "canonical_emission.hpp"
#include "cone_traits.hpp"
#include "convex_traits.hpp"

namespace convex {

/**
 * @brief Inequality
 *
 * lhs <= rhs... 0 <= rhs - lhs (rplus)
 * lhs >= rhs... 0 <= lhs - rhs (rplus)
 * lhs == rhs... affine equality row lhs - rhs = 0
 */
template <InequalityType IType, AtomLike LHS, typename RHS>
struct Inequality {
  /// @brief extract alias & types from base
  using Meta = atom_meta_t<LHS>;
  using Domain = meta_dom_t<Meta>;
  using Codom = meta_codom_t<Meta>;
  using T = dom_type_t<Domain>;
  using Point = dom_point_t<Domain>;
  static constexpr std::size_t M = dom_size_v<Codom>;

  static_assert(is_fn_type_v<LHS> || is_index_type_v<LHS> || is_var_type_v<LHS>,
                "LHS Atom must be fn, index, or var type");

  /**
   * @brief CTOR
   *
   * @param lhs (Atom / Scalar)
   * @param rhs (Atom / Scalar)
   */
  constexpr explicit Inequality(const LHS& lhs_in, const RHS& rhs_in)
      : lhs(lhs_in), rhs(rhs_in) {}

  /**
   * @brief canonicalize
   *
   * Follows DCP convention:
   * LEQ: lhs - rhs <= 0, encoded as 0 <= rhs - lhs in the nonnegative cone
   * GEQ: lhs - rhs >= 0, encoded directly in the nonnegative cone
   * -> Works directly on cone map F & g
   *
   * EQ: lhs - rhs == 0 (i.e., convex - (concave))
   * -> Works directly on A & b affine equality map
   *
   * @param builder
   *
   * @return new builder
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize(const Builder& builder) const {
    auto canonicalize_atom = [&](const auto& atom, const auto& parent_builder) {
      using AtomT = linalg::remove_cvref_t<decltype(atom)>;
      if constexpr (is_var_type_v<AtomT>)
        return atom.template canonicalize<AtomType::VAR>(parent_builder);
      else return atom.template canonicalize<AtomType::FN>(parent_builder);
    };

    auto negate_atom = [](const auto& atom) {
      using AtomT = linalg::remove_cvref_t<decltype(atom)>;
      if constexpr (is_index_type_v<AtomT> || is_var_type_v<AtomT>) {
        return atom.mul(T(-1));
      } else {
        return atom.template mul<T(-1)>();
      }
    };

    auto make_broadcast = [](T scalar) {
      linalg::Vector<T, M> out{};
      for (std::size_t i = 0; i < M; ++i) out(i) = scalar;
      return out;
    };

    auto append_equality = [](const auto& child_builder, const auto& eq_linear,
                              const auto& eq_bias) {
      const auto eq_map =
          affine::AffineMap<decltype(eq_linear)>(eq_linear, eq_bias);
      const auto emission = conic::LocalEqualityEmission(eq_map);
      return child_builder.realize(emission);
    };

    if constexpr (dom_vartype_v<Codom> == VarType::PSD &&
                  linalg::MatrixContainer<RHS>) {
      static_assert(is_var_type_v<LHS>,
                    "PSD inequalities only support variable lhs");
      static_assert(IType == InequalityType::GEQ,
                    "PSD inequalities currently only support X >= C");

      using CodomPoint = dom_point_t<Codom>;
      using RHSMat = linalg::remove_cvref_t<RHS>;
      using RHSScalar = linalg::matrix_scalar_t<RHSMat>;
      static_assert(linalg::mat_rows_v<RHSMat> ==
                        linalg::mat_rows_v<CodomPoint>,
                    "PSD inequality rhs row dim mismatch");
      static_assert(linalg::mat_cols_v<RHSMat> ==
                        linalg::mat_cols_v<CodomPoint>,
                    "PSD inequality rhs col dim mismatch");
      static_assert(std::convertible_to<RHSScalar, T>,
                    "PSD inequality rhs scalar type must convert to lhs type");

      const auto child_builder = canonicalize_atom(lhs, builder);
      const auto child_coord = child_builder.coord_map;

      static constexpr std::size_t MatDim = linalg::mat_rows_v<CodomPoint>;
      static constexpr std::size_t K = MatDim * (MatDim + 1) / 2;

      if constexpr (linalg::Real<RHSScalar>)
        AK_ASSERT(linalg::all_equal(rhs, linalg::transpose(rhs)) &&
                  "PSD inequality expects symmetric matrix rhs");
      else
        AK_ASSERT(linalg::all_equal(rhs, linalg::conjugate_transpose(rhs)) &&
                  "PSD inequality expects Hermitian matrix rhs");

      const auto F = child_coord.linear_part();
      const auto g = child_coord.translation() - linalg::get_svec(rhs);
      const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
      const auto psd_cone =
          conic::PositiveSemidefiniteCone<linalg::Vector<T, K>, MatDim>();
      const auto cones = std::make_tuple(psd_cone);
      const auto emission =
          conic::LocalEmission(coord_map_up, cones, child_builder.eq_map);
      return child_builder.template realize<AtomType::CAP>(emission);
    } else if constexpr (linalg::Field<RHS>) {
      const auto child_builder = canonicalize_atom(lhs, builder);
      const auto child_coord = child_builder.coord_map;

      using ChildMat = decltype(child_coord.linear_part());
      using CodomPoint = dom_point_t<Codom>;

      if constexpr (dom_vartype_v<Codom> == VarType::PSD) {
        static_assert(is_var_type_v<LHS>,
                      "PSD inequalities only support variable lhs");
        static_assert(IType == InequalityType::GEQ,
                      "PSD inequalities currently only support X >= 0");
        AK_ASSERT(rhs == T(0) && "PSD inequality expects zero rhs");

        static constexpr std::size_t MatDim = linalg::mat_rows_v<CodomPoint>;
        static constexpr std::size_t K = MatDim * (MatDim + 1) / 2;

        const auto F = child_coord.linear_part();
        const auto g = child_coord.translation();
        const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
        const auto psd_cone =
            conic::PositiveSemidefiniteCone<linalg::Vector<T, K>, MatDim>();
        const auto cones = std::make_tuple(psd_cone);
        const auto emission =
            conic::LocalEmission(coord_map_up, cones, child_builder.eq_map);
        return child_builder.template realize<AtomType::CAP>(emission);
      } else {
        if constexpr (IType == InequalityType::LEQ) {
          ChildMat F{T(-1) * child_coord.linear_part()};
          linalg::Vector<T, M> g{make_broadcast(rhs) -
                                 child_coord.translation()};
          const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
          const auto cones =
              std::make_tuple(conic::NonnegOrthantCone<linalg::Vector<T, M>>());
          const auto emission =
              conic::LocalEmission(coord_map_up, cones, child_builder.eq_map);
          return child_builder.template realize<AtomType::CAP>(emission);
        } else if constexpr (IType == InequalityType::GEQ) {
          ChildMat F{child_coord.linear_part()};
          linalg::Vector<T, M> g{child_coord.translation() -
                                 make_broadcast(rhs)};
          const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
          const auto cones =
              std::make_tuple(conic::NonnegOrthantCone<linalg::Vector<T, M>>());
          const auto emission =
              conic::LocalEmission(coord_map_up, cones, child_builder.eq_map);
          return child_builder.template realize<AtomType::CAP>(emission);
        } else {
          const auto A_eq = child_coord.linear_part();
          linalg::Vector<T, M> b_eq{make_broadcast(rhs) -
                                    child_coord.translation()};
          return append_equality(child_builder, A_eq, b_eq);
        }
      }
    } else {
      if constexpr (!linalg::MatrixContainer<RHS>) {
        static_assert(
            is_fn_type_v<RHS> || is_index_type_v<RHS> || is_var_type_v<RHS>,
            "RHS Atom must be fn, index, var, scalar, or matrix type");
      }

      if constexpr (IType == InequalityType::LEQ) {
        const auto new_lhs = negate_atom(lhs);
        const Sum sum(new_lhs, rhs);
        const auto new_builder =
            sum.template canonicalize<AtomType::ADD>(builder);
        const auto cones =
            std::make_tuple(conic::NonnegOrthantCone<linalg::Vector<T, 1>>());
        const auto emission = conic::LocalEmission(new_builder.coord_map, cones,
                                                   new_builder.eq_map);
        return new_builder.template realize<AtomType::CAP>(emission);
      } else if constexpr (IType == InequalityType::GEQ) {
        const auto new_rhs = negate_atom(rhs);
        const Sum sum(lhs, new_rhs);
        const auto new_builder =
            sum.template canonicalize<AtomType::ADD>(builder);
        const auto cones =
            std::make_tuple(conic::NonnegOrthantCone<linalg::Vector<T, 1>>());
        const auto emission = conic::LocalEmission(new_builder.coord_map, cones,
                                                   new_builder.eq_map);
        return new_builder.template realize<AtomType::CAP>(emission);
      } else {
        const auto new_rhs = negate_atom(rhs);
        const Sum sum(lhs, new_rhs);
        const auto child_builder =
            sum.template canonicalize<AtomType::ADD>(builder);
        const auto child_coord = child_builder.coord_map;
        const auto A_eq = child_coord.linear_part();
        linalg::Vector<T, 1> b_eq{child_coord.translation()(0)};
        return append_equality(child_builder, A_eq, b_eq);
      }
    }
  }

  /** @brief member vars */
  LHS lhs; ///< LHS atom
  RHS rhs; ///< RHS atom
};

} // namespace convex
