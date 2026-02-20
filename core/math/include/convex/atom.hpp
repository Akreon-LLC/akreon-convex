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

#include "config/assert.hpp"

#include <cassert>

#include "convex_traits.hpp"
#include "convex_util.hpp"

namespace convex {

template <typename Meta, typename Derived, typename... Cs>
struct Atom {
  using Domain = meta_dom_t<Meta>;
  using Codomain = meta_codom_t<Meta>;
  using T = dom_type_t<Domain>;
  using PointT = dom_point_t<Domain>;

  static constexpr size_t N = dom_size_v<Domain>;
  static constexpr size_t M = dom_size_v<Codomain>;
  static constexpr CurvatureType Curvature = meta_curv_v<Meta>;
  static constexpr SignType DomSign = dom_sign_v<Domain>;
  static constexpr SignType CodomSign = dom_sign_v<Codomain>;

  constexpr explicit Atom() { validate_domain(); }

  constexpr explicit Atom(const std::tuple<Cs...> child_in)
      : children(std::move(child_in)) {
    validate_domain();
  }

  /**
   * @brief validate domain for fn
   *
   * @param child curvature
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

  constexpr void validate_domain() const {
    size_t num_convex = 0, num_concave = 0, num_affine = 0;
    auto validate_child = [&](const auto& child) {
      if constexpr (is_fn_type_v<decltype(self())>) {
        validate_fn_domain(child);
      }

      if constexpr (is_set_type_v<decltype(self())>) {
        static_assert(is_var_type_v<decltype(child)>,
                      "Set expects child to be a variable type");
      }

      const SignType& dom_sign =
          dom_sign_v<meta_dom_t<atom_meta_t<decltype(self())>>>;
      if (dom_sign != SignType::FREE) {
        const SignType& child_sign =
            dom_sign_v<meta_codom_t<atom_meta_t<decltype(child)>>>;
        switch (dom_sign) {
          case SignType::POS: {
            AK_ASSERT(child_sign == SignType::POS &&
                   "Parent positive domain expects positive child codomain");
            break;
          }
          case SignType::NONNEG: {
            AK_ASSERT((child_sign == SignType::POS ||
                    child_sign == SignType::NONNEG) &&
                   "Parent non-negative domain expects positive/non-negative child codomain");
            break;
          }
          case SignType::NEG: {
            AK_ASSERT(child_sign == SignType::NEG &&
                   "Parent negative domain expects negative child codomain");
            break;
          }
          case SignType::NONPOS: {
            AK_ASSERT((child_sign == SignType::NONPOS ||
                    child_sign == SignType::NEG) &&
                   "Parent non-positive domain expects non-positive/negative child codomain");
            break;
          }
          case SignType::PD: {
            AK_ASSERT(child_sign == SignType::PD &&
                   "Parent PD domain expects PD child codomain");
            break;
          }
          case SignType::PSD: {
            AK_ASSERT((child_sign == SignType::PSD || child_sign == SignType::PD) &&
                   "Parent PSD domain expects PSD/PD child codomain");
            break;
          }
          case SignType::ND: {
            AK_ASSERT(child_sign == SignType::ND &&
                   "Parent ND domain expects ND child codomain");
            break;
          }
          case SignType::NSD: {
            AK_ASSERT((child_sign == SignType::NSD || child_sign == SignType::ND) &&
                   "Parent NSD domain expects NSD/ND child codomain");
            break;
          }
          case SignType::FREE:
          default:
            break;
        }
        (void)child_sign;
      }

      if constexpr (is_cap_type_v<decltype(self())>) {
        AK_ASSERT(is_set_type_v<decltype(child)> &&
               "Intersection expects each child to be a set");
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

  template <AtomType Type, BuilderLike Builder>
  constexpr auto canonicalize(const Builder& builder) const {
    return self().template canonicalize_impl<Type>(builder);
  }

  constexpr const Derived& self() const {
    return static_cast<const Derived&>(*this);
  }

  std::tuple<Cs...> children;
};

}  // namespace convex
