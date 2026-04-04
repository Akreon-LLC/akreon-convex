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
  static constexpr size_t M = dom_size_v<Codom>;

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
   * LEQ: lhs - rhs <= 0 (i.e., convex - (concave))
   * GEQ: 0 <= rhs - lhs (i.e., convex - (concave))
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
      else
        return atom.template canonicalize<AtomType::FN>(parent_builder);
    };

    auto negate_atom = [](const auto& atom) {
      using AtomT = linalg::remove_cvref_t<decltype(atom)>;
      if constexpr (is_index_type_v<AtomT> || is_var_type_v<AtomT>) {
        return atom.mul(T(-1));
      } else {
        return atom.template mul<T(-1)>();
      }
    };

    auto make_broadcast = [](const T& scalar) {
      linalg::Vector<T, M> out{};
      for (size_t i = 0; i < M; ++i) out(i) = scalar;
      return out;
    };

    auto append_equality = [](const auto& child_builder, const auto& eq_linear,
                              const auto& eq_bias) {
      const auto child_cb = child_builder.cb;
      const auto child_eq = child_builder.eq_map;

      using EqMat = std::decay_t<decltype(eq_linear)>;
      using ChildEqMat = affine::affine_mat_t<decltype(child_eq)>;

      static constexpr size_t NewRows = linalg::mat_rows_v<EqMat>;
      static constexpr size_t NewCols = linalg::mat_cols_v<EqMat>;
      static constexpr size_t ChildEqRows = linalg::mat_rows_v<ChildEqMat>;
      static constexpr size_t ChildEqCols = linalg::mat_cols_v<ChildEqMat>;
      static constexpr size_t FullCols =
          (ChildEqCols > NewCols) ? ChildEqCols : NewCols;
      static constexpr size_t FullRows = ChildEqRows + NewRows;

      linalg::Matrix<T, FullRows, FullCols> A_eq{};
      linalg::Vector<T, FullRows> b_eq{};

      if constexpr (ChildEqRows != 0 && ChildEqCols != 0) {
        A_eq = linalg::set_block<0, ChildEqRows, 0, ChildEqCols>(
            A_eq, child_eq.linear_part());
        for (size_t i = 0; i < ChildEqRows; ++i)
          b_eq(i) = child_eq.translation()(i);
      }

      A_eq =
          linalg::set_block<ChildEqRows, FullRows, 0, NewCols>(A_eq, eq_linear);
      for (size_t i = 0; i < NewRows; ++i) b_eq(i + ChildEqRows) = eq_bias(i);

      const auto new_eq_map = affine::AffineMap<decltype(A_eq)>(A_eq, b_eq);
      return conic::CanonicalFormBuilder<
          conic::builder_dom_t<Builder>, decltype(child_builder.coord_map),
          decltype(child_cb), decltype(new_eq_map)>(child_builder.coord_map,
                                                    child_cb, new_eq_map);
    };

    if constexpr (dom_vartype_v<Codom> == VarType::PSD &&
                  linalg::MatrixLike<linalg::remove_cvref_t<RHS>>) {
      static_assert(is_var_type_v<LHS>,
                    "PSD inequalities only support variable lhs");
      static_assert(IType == InequalityType::GEQ,
                    "PSD inequalities currently only support X >= C");

      using CodomPoint = dom_point_t<Codom>;
      using RHSMat = linalg::remove_cvref_t<RHS>;
      using RHSScalar = linalg::mat_type_t<RHSMat>;
      static_assert(
          linalg::mat_rows_v<RHSMat> == linalg::mat_rows_v<CodomPoint>,
          "PSD inequality rhs row dim mismatch");
      static_assert(
          linalg::mat_cols_v<RHSMat> == linalg::mat_cols_v<CodomPoint>,
          "PSD inequality rhs col dim mismatch");
      static_assert(std::convertible_to<RHSScalar, T>,
                    "PSD inequality rhs scalar type must convert to lhs type");

      const auto child_builder = canonicalize_atom(lhs, builder);
      const auto child_coord = child_builder.coord_map;
      const auto child_cb = child_builder.cb;

      static constexpr size_t MatDim = linalg::mat_rows_v<CodomPoint>;
      static constexpr size_t K = MatDim * (MatDim + 1) / 2;

      if constexpr (linalg::RealLike<RHSScalar>)
        AK_ASSERT(linalg::all_equal(rhs, linalg::transpose(rhs)) &&
                  "PSD inequality expects symmetric matrix rhs");
      else
        AK_ASSERT(linalg::all_equal(rhs, linalg::conjugate_transpose(rhs)) &&
                  "PSD inequality expects Hermitian matrix rhs");

      linalg::Matrix<T, M, M> S{};
      size_t idx{0};
      for (size_t i = 0; i < MatDim; ++i)
        for (size_t j = 0; j <= i; ++j, ++idx)
          S(idx)(idx) = (i == j) ? T(1) : linalg::constexpr_sqrt_safe(T(2));

      /// (dakre) svec <-> smat is a linear isomorphism between S^n and
      /// R^{n(n+1)/2}
      /// 1. express the symmetric matrix var in psd-cone coords via svec
      /// (off-diag scaling by sqrt(2))
      /// 2. translate the affine bias by -svec(rhs) so the cone sees X - rhs
      /// 3. enforce membership in the psd cone (i.e., smat(F z + g) >= 0)
      const auto F = S * child_coord.linear_part();
      const auto g = S * child_coord.translation() - linalg::get_svec(rhs);
      const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
      const auto new_builder = conic::CanonicalFormBuilder<
          conic::builder_dom_t<decltype(builder)>, decltype(coord_map_up),
          decltype(child_cb), decltype(child_builder.eq_map)>(
          coord_map_up, child_cb, child_builder.eq_map);
      const auto psd_cone =
          conic::PositiveSemidefiniteCone<linalg::Vector<T, K>, MatDim>();
      return new_builder.template make<AtomType::CAP>(
          std::make_tuple(psd_cone));
    } else if constexpr (linalg::FieldLike<RHS>) {
      const auto child_builder = canonicalize_atom(lhs, builder);
      const auto child_coord = child_builder.coord_map;
      const auto child_cb = child_builder.cb;

      using ChildMat = decltype(child_coord.linear_part());
      using CodomPoint = dom_point_t<Codom>;

      if constexpr (dom_vartype_v<Codom> == VarType::PSD) {
        static_assert(is_var_type_v<LHS>,
                      "PSD inequalities only support variable lhs");
        static_assert(IType == InequalityType::GEQ,
                      "PSD inequalities currently only support X >= 0");
        AK_ASSERT(rhs == T(0) && "PSD inequality expects zero rhs");

        static constexpr size_t MatDim = linalg::mat_rows_v<CodomPoint>;
        static constexpr size_t K = MatDim * (MatDim + 1) / 2;

        linalg::Matrix<T, M, M> S{};
        size_t idx{0};
        for (size_t i = 0; i < MatDim; ++i)
          for (size_t j = 0; j <= i; ++j, ++idx)
            S(idx)(idx) = (i == j) ? T(1) : linalg::constexpr_sqrt_safe(T(2));

        /// (dakre) svec <-> smat is a linear isomorphism between S^n and
        /// R^{n(n+1)/2}
        /// 1. express the symmetric matrix var in psd-cone coords via svec
        /// (off-diag scaling by sqrt(2))
        /// 2. enforce membership in the psd cone (i.e., smat(F z + g) >= 0)
        const auto F = S * child_coord.linear_part();
        const auto g = S * child_coord.translation();
        const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
        const auto new_builder = conic::CanonicalFormBuilder<
            conic::builder_dom_t<decltype(builder)>, decltype(coord_map_up),
            decltype(child_cb), decltype(child_builder.eq_map)>(
            coord_map_up, child_cb, child_builder.eq_map);
        const auto psd_cone =
            conic::PositiveSemidefiniteCone<linalg::Vector<T, K>, MatDim>();
        return new_builder.template make<AtomType::CAP>(
            std::make_tuple(psd_cone));
      } else {
        if constexpr (IType == InequalityType::LEQ) {
          ChildMat F{T(-1) * child_coord.linear_part()};
          linalg::Vector<T, M> g{make_broadcast(rhs) -
                                 child_coord.translation()};
          const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
          const auto new_builder = conic::CanonicalFormBuilder<
              conic::builder_dom_t<decltype(builder)>, decltype(coord_map_up),
              decltype(child_cb), decltype(child_builder.eq_map)>(
              coord_map_up, child_cb, child_builder.eq_map);
          return new_builder.template make<AtomType::CAP>(std::make_tuple(
              conic::NonnegOrthantCone<linalg::Vector<T, M>>()));
        } else if constexpr (IType == InequalityType::GEQ) {
          ChildMat F{child_coord.linear_part()};
          linalg::Vector<T, M> g{child_coord.translation() -
                                 make_broadcast(rhs)};
          const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
          const auto new_builder = conic::CanonicalFormBuilder<
              conic::builder_dom_t<decltype(builder)>, decltype(coord_map_up),
              decltype(child_cb), decltype(child_builder.eq_map)>(
              coord_map_up, child_cb, child_builder.eq_map);
          return new_builder.template make<AtomType::CAP>(std::make_tuple(
              conic::NonnegOrthantCone<linalg::Vector<T, M>>()));
        } else {
          const auto A_eq = child_coord.linear_part();
          linalg::Vector<T, M> b_eq{make_broadcast(rhs) -
                                    child_coord.translation()};
          return append_equality(child_builder, A_eq, b_eq);
        }
      }
    } else {
      if constexpr (!linalg::MatrixLike<linalg::remove_cvref_t<RHS>>) {
        static_assert(
            is_fn_type_v<RHS> || is_index_type_v<RHS> || is_var_type_v<RHS>,
            "RHS Atom must be fn, index, var, scalar, or matrix type");
      }

      if constexpr (IType == InequalityType::LEQ) {
        const auto new_lhs = negate_atom(lhs);
        const Sum sum(new_lhs, rhs);
        const auto new_builder =
            sum.template canonicalize<AtomType::ADD>(builder);
        return new_builder.template make<AtomType::CAP>(
            std::make_tuple(conic::NonnegOrthantCone<linalg::Vector<T, 1>>()));
      } else if constexpr (IType == InequalityType::GEQ) {
        const auto new_rhs = negate_atom(rhs);
        const Sum sum(lhs, new_rhs);
        const auto new_builder =
            sum.template canonicalize<AtomType::ADD>(builder);
        return new_builder.template make<AtomType::CAP>(
            std::make_tuple(conic::NonnegOrthantCone<linalg::Vector<T, 1>>()));
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
  LHS lhs;  ///< LHS atom
  RHS rhs;  ///< RHS atom
};

}  // namespace convex
