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
 * lhs == rhs... 0 == lhs - rhs (zoc)
 */
template <InequalityType IType, AtomLike LHS, typename RHS>
struct Inequality {
  /// @brief extract alias & types from base
  using Meta = atom_meta_t<LHS>;
  using Domain = meta_dom_t<Meta>;
  using Codom = meta_codom_t<Meta>;
  using T = dom_type_t<Domain>;
  using Point = dom_point_t<Domain>;

  static_assert(is_fn_type_v<LHS> || is_index_type_v<LHS>,
                "LHS Atom must be fn or index type");

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
   * @param builder
   *
   * @return new builder
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize(const Builder& builder) const {
    auto get_cone = []() {
      if constexpr (IType == InequalityType::EQ) {
        return conic::ZeroOrderCone<linalg::Vector<T, 1>>();
      } else {
        return conic::NonnegOrthantCone<linalg::Vector<T, 1>>();
      }
    };

    if constexpr (linalg::FieldLike<RHS>) {
      const auto child_builder =
          lhs.template canonicalize<AtomType::FN>(builder);
      const auto child_coord = child_builder.coord_map;
      const auto child_cb = child_builder.cb;

      using ChildMat = decltype(child_coord.linear_part());

      ChildMat F{child_coord.linear_part()};
      linalg::Vector<T, 1> g{child_coord.translation()};
      if constexpr (IType == InequalityType::LEQ) {
        F = T(-1) * F;
        g = linalg::Vector<T, 1>{rhs} - g;
      } else {
        g = g - linalg::Vector<T, 1>{rhs};
      }

      const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
      const auto new_builder =
          conic::ConeBlockBuilder<conic::builder_dom_t<decltype(builder)>,
                                  decltype(coord_map_up), decltype(child_cb)>(
              coord_map_up, child_cb);

      return new_builder.template make<AtomType::CAP>(
          std::make_tuple(get_cone()));
    } else {
      static_assert(is_fn_type_v<RHS> || is_index_type_v<LHS>,
                    "RHS Atom must be fn, index, or scalar type");

      auto negate_atom = [](const auto& atom) {
        if constexpr (is_index_type_v<decltype(atom)>) {
          return atom.mul(T(-1));
        } else {
          return atom.template mul<T(-1)>();
        }
      };

      using SumDom = linalg::Matrix<T, 1, 2>;
      using SumCodom = linalg::Matrix<T, 1, 1>;

      if constexpr (IType == InequalityType::LEQ) {
        const auto new_lhs = negate_atom(lhs);
        const Sum<SumMeta<SumDom, SumCodom>, decltype(new_lhs), decltype(rhs)>
            sum(new_lhs, rhs);
        const auto new_builder =
            sum.template canonicalize<AtomType::ADD>(builder);
        return new_builder.template make<AtomType::CAP>(
            std::make_tuple(get_cone()));
      } else {
        const auto new_rhs = negate_atom(rhs);
        const Sum<SumMeta<SumDom, SumCodom>, decltype(lhs), decltype(new_rhs)>
            sum(lhs, new_rhs);
        const auto new_builder =
            sum.template canonicalize<AtomType::ADD>(builder);
        return new_builder.template make<AtomType::CAP>(
            std::make_tuple(get_cone()));
      }
    }
  }

  /** @brief member vars */
  LHS lhs;  ///< LHS atom
  RHS rhs;  ///< RHS atom
};

}  // namespace convex
