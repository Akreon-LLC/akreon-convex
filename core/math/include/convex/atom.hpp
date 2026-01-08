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
    if constexpr (dom_size_v<ParentDom> != dom_size_v<ChildCodom>) {
      throw std::domain_error(
          "Mismatch in size between parent domain and child codomain");
    }
    const CurvatureType& child_curv = meta_curv_v<atom_meta_t<decltype(child)>>;
    const CurvatureType parent_curv = meta_curv_v<Meta>;
    if (self().monotonicity() == MonotonicityType::UNKNOWN &&
        child_curv != CurvatureType::AFFINE)
      throw std::domain_error(
          "Function's monotonicity is unknown and child curvature is "
          "non-affine");

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

    throw std::domain_error("Functions composition fails DCP compliance");
  }

  constexpr void validate_domain() const {
    size_t num_convex = 0, num_concave = 0;
    auto validate_child = [&](const auto& child) {
      if constexpr (is_fn_type_v<decltype(self())>) {
        validate_fn_domain(child);
      }

      if constexpr (is_set_type_v<decltype(self())>) {
        if (!is_var_type_v<decltype(child)>)
          throw std::domain_error("Set expects child to be a variable type");
      }

      const SignType& dom_sign =
          dom_sign_v<meta_dom_t<atom_meta_t<decltype(self())>>>;
      if (dom_sign != SignType::FREE) {
        const SignType& child_sign =
            dom_sign_v<meta_codom_t<atom_meta_t<decltype(child)>>>;
        switch (dom_sign) {
          case SignType::POS: {
            if (child_sign != SignType::POS)
              throw std::domain_error("Parent expects positive child codomain");
            break;
          }
          case SignType::NONNEG: {
            if (child_sign != SignType::POS && child_sign != SignType::NONNEG)
              throw std::domain_error(
                  "Parent expects non-negative child codomain");
            break;
          }
          case SignType::NEG: {
            if (child_sign != SignType::NEG)
              throw std::domain_error("Parent expects negative child codomain");
            break;
          }
          case SignType::NONPOS: {
            if (child_sign != SignType::NONPOS && child_sign != SignType::NEG)
              throw std::domain_error(
                  "Parent expects non-positive child codomain");
            break;
          }
          case SignType::PD: {
            if (child_sign != SignType::PD)
              throw std::domain_error(
                  "Parent expects positive definite child codomain");
            break;
          }
          case SignType::PSD: {
            if (child_sign != SignType::PSD && child_sign != SignType::PD)
              throw std::domain_error(
                  "Parent expects positive semi-definite child codomain");
            break;
          }
          case SignType::ND: {
            if (child_sign != SignType::ND)
              throw std::domain_error(
                  "Parent expects negative definite child codomain");
            break;
          }
          case SignType::NSD: {
            if (child_sign != SignType::NSD && child_sign != SignType::ND)
              throw std::domain_error(
                  "Parent expects negative semi-definite child codomain");
            break;
          }
          case SignType::FREE:
          default:
            break;
        }
      }

      if constexpr (is_cap_type_v<decltype(self())>) {
        if (!is_set_type_v<decltype(child)>)
          throw std::domain_error("Intersection operating on non-convex set");
      }

      if constexpr (is_add_type_v<decltype(self())>) {
        /// (dakre) TODO add in minkowski sum
        if (is_set_type_v<decltype(child)>)
          throw std::domain_error("Minkowski sum is currently not supported");
      }

      const CurvatureType& child_curv =
          meta_curv_v<atom_meta_t<decltype(child)>>;
      num_convex += child_curv != CurvatureType::CONCAVE;
      num_concave += child_curv != CurvatureType::CONVEX;
    };
    std::apply([&](const auto&... child) { (validate_child(child), ...); },
               children);

    if constexpr (is_add_type_v<decltype(self())>) {
      const CurvatureType& parent_curv = meta_curv_v<Meta>;
      if (parent_curv == CurvatureType::CONVEX) {
        if (num_convex != N)
          throw std::domain_error("Parent domain is expecting convex children");
      } else if (parent_curv == CurvatureType::CONCAVE) {
        if (num_concave != N)
          throw std::domain_error(
              "Parent domain is expecting concave children");
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
