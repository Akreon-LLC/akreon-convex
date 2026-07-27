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

#include "canonical_form_builder.hpp"
#include "config/assert.hpp"
#include "convex_traits.hpp"

namespace convex {

/**
 * @brief CRTP base for convex atoms.
 *
 * The base stores normalized child atoms and enforces domain, codomain,
 * curvature, and DCP composition rules shared by functions, sets,
 * intersections, and sums.
 */
template <typename Meta, typename Derived, typename... Cs>
struct Atom {
  /// @brief Atom input domain.
  using Domain = meta_dom_t<Meta>;
  /// @brief Atom output codomain.
  using Codomain = meta_codom_t<Meta>;
  /// @brief Tuple of decayed child atom types.
  using ChildrenT = std::tuple<atom_decay_t<Cs>...>;
  /// @brief Scalar field for the atom domain.
  using T = dom_type_t<Domain>;
  /// @brief Point type accepted by the atom domain.
  using PointT = dom_point_t<Domain>;

  /// @brief Domain dimension.
  static constexpr std::size_t N = dom_size_v<Domain>;
  /// @brief Codomain dimension.
  static constexpr std::size_t M = dom_size_v<Codomain>;
  /// @brief Curvature declared by atom metadata.
  static constexpr CurvatureType Curvature = meta_curv_v<Meta>;
  /// @brief Domain sign declared by atom metadata.
  static constexpr VarType DomSign = dom_vartype_v<Domain>;
  /// @brief Codomain sign declared by atom metadata.
  static constexpr VarType CodomSign = dom_vartype_v<Codomain>;

  /**
   * @brief Construct a leaf atom with no children.
   */
  constexpr explicit Atom() { validate_domain(); }

  /**
   * @brief Construct an atom with normalized children.
   *
   * @param child_in tuple of child atoms
   */
  constexpr explicit Atom(ChildrenT child_in) : children(std::move(child_in)) {
    validate_domain();
  }

  /**
   * @brief Validate DCP composition for a function child.
   *
   * @param child child atom whose codomain feeds this function's domain
   *
   * @return void
   */
  constexpr void validate_fn_domain(const auto& child) const {
    using ChildCodom = meta_codom_t<atom_meta_t<decltype(child)>>;
    using ParentDom = meta_dom_t<Meta>;
    static_assert(dom_size_v<ParentDom> == dom_size_v<ChildCodom>,
                  "Mismatch in size between parent domain and child codomain");

    const CurvatureType& child_curv = meta_curv_v<atom_meta_t<decltype(child)>>;
    const CurvatureType parent_curv = meta_curv_v<Meta>;

    AK_ASSERT((self().monotonicity() != MonotonicityType::UNKNOWN ||
               child_curv == CurvatureType::AFFINE) &&
              "Function with unknown monotonicity requires affine child");

    if (child_curv == CurvatureType::AFFINE ||
        parent_curv == CurvatureType::AFFINE)
      return;

    if (parent_curv == CurvatureType::CONVEX) {
      if ((self().monotonicity() == MonotonicityType::NONDEC &&
           child_curv == CurvatureType::CONVEX) ||
          (self().monotonicity() == MonotonicityType::NONINC &&
           child_curv == CurvatureType::CONCAVE))
        return;
    }

    if (parent_curv == CurvatureType::CONCAVE) {
      if ((self().monotonicity() == MonotonicityType::NONDEC &&
           child_curv == CurvatureType::CONCAVE) ||
          (self().monotonicity() == MonotonicityType::NONINC &&
           child_curv == CurvatureType::CONVEX))
        return;
    }

    const bool dcp_composition_valid =
        (parent_curv == CurvatureType::CONVEX &&
         ((self().monotonicity() == MonotonicityType::NONDEC &&
           child_curv == CurvatureType::CONVEX) ||
          (self().monotonicity() == MonotonicityType::NONINC &&
           child_curv == CurvatureType::CONCAVE))) ||
        (parent_curv == CurvatureType::CONCAVE &&
         ((self().monotonicity() == MonotonicityType::NONDEC &&
           child_curv == CurvatureType::CONCAVE) ||
          (self().monotonicity() == MonotonicityType::NONINC &&
           child_curv == CurvatureType::CONVEX)));
    AK_ASSERT(dcp_composition_valid &&
              "Function composition fails DCP compliance");
    (void)dcp_composition_valid;
  }

  /**
   * @brief Validate child domains, signs, curvature, and composition rules.
   *
   * @return void
   */
  constexpr void validate_domain() const {
    std::size_t num_convex = 0, num_concave = 0, num_affine = 0;
    auto validate_child = [&](const auto& child) {
      if constexpr (is_fn_type_v<decltype(self())>) {
        validate_fn_domain(child);
      }

      if constexpr (is_set_type_v<decltype(self())>) {
        static_assert(is_var_type_v<decltype(child)>,
                      "Set expects child to be a variable type");
      }

      const VarType& dom_sign =
          dom_vartype_v<meta_dom_t<atom_meta_t<decltype(self())>>>;
      if (dom_sign != VarType::FREE) {
        const VarType& child_sign =
            dom_vartype_v<meta_codom_t<atom_meta_t<decltype(child)>>>;
        switch (dom_sign) {
          case VarType::POS: {
            AK_ASSERT(child_sign == VarType::POS &&
                      "Parent positive domain expects positive child codomain");
            break;
          }
          case VarType::NONNEG: {
            AK_ASSERT(
                (child_sign == VarType::POS || child_sign == VarType::NONNEG) &&
                "Parent non-negative domain expects "
                "positive/non-negative child codomain");
            break;
          }
          case VarType::NEG: {
            AK_ASSERT(child_sign == VarType::NEG &&
                      "Parent negative domain expects negative child codomain");
            break;
          }
          case VarType::NONPOS: {
            AK_ASSERT(
                (child_sign == VarType::NONPOS || child_sign == VarType::NEG) &&
                "Parent non-positive domain expects "
                "non-positive/negative child codomain");
            break;
          }
          case VarType::PSD: {
            AK_ASSERT(child_sign == VarType::PSD &&
                      "Parent PSD domain expects PSD child codomain");
            break;
          }
          case VarType::FREE:
          default:
            break;
        }
        (void)child_sign;
      }

      if constexpr (is_cap_type_v<decltype(self())>) {
        AK_ASSERT((is_set_type_v<decltype(child)> ||
                   is_cap_type_v<decltype(child)>) &&
                  "Intersection expects each child to be a set or "
                  "intersection");
      }

      if constexpr (is_add_type_v<decltype(self())>) {
        /// (dakre) TODO add in minkowski sum
        AK_ASSERT(!is_set_type_v<decltype(child)> &&
                  "Minkowski sum is not supported for set children");
      }

      const CurvatureType& child_curv =
          meta_curv_v<atom_meta_t<decltype(child)>>;
      num_affine += child_curv == CurvatureType::AFFINE;
      num_convex += child_curv == CurvatureType::CONVEX;
      num_concave += child_curv == CurvatureType::CONCAVE;
    };
    std::apply([&](const auto&... child) { (validate_child(child), ...); },
               children);

    if constexpr (is_add_type_v<decltype(self())>) {
      if ((num_convex == N || num_convex + num_affine == N) &&
          num_concave == 0) {
        // no-op - valid dcp
      } else if ((num_concave == N || num_concave + num_affine == N) &&
                 num_convex == 0) {
        // no-op - valid dcp
      } else {
        const bool dcp_sum_valid =
            ((num_convex == N || num_convex + num_affine == N) &&
             num_concave == 0) ||
            ((num_concave == N || num_concave + num_affine == N) &&
             num_convex == 0);
        AK_ASSERT(dcp_sum_valid &&
                  "Not dcp valid to sum convex with concave atoms");
        (void)dcp_sum_valid;
      }
    }
  }

  /**
   * @brief Dispatch canonicalization to the derived atom.
   *
   * @tparam Type requested atom canonicalization target
   * @tparam Builder canonical-form builder type
   *
   * @param builder current canonical-form builder
   *
   * @return derived atom canonicalization result
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize(const Builder& builder) const {
    return self().template canonicalize_impl<Type>(builder);
  }

  /**
   * @brief Return this object as the CRTP-derived atom type.
   *
   * @return derived object
   */
  constexpr const Derived& self() const {
    return static_cast<const Derived&>(*this);
  }

  ChildrenT children; ///< Tuple of normalized child atoms.
};

} // namespace convex
