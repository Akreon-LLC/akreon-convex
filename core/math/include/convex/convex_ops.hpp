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

#include <optional>

#include "convex_traits.hpp"

namespace convex {

template <typename Meta, typename... Cs>
struct Sum : Atom<Meta, Sum<Meta, Cs...>, Cs...> {
  using BaseT = Atom<Meta, Sum<Meta, Cs...>, Cs...>;
  using typename BaseT::Domain;
  using typename BaseT::PointT;
  using typename BaseT::T;
  static_assert(dom_size_v<Domain> == sizeof...(Cs));

  constexpr explicit Sum(const Cs... cs)
      : BaseT(std::make_tuple(std::forward<Cs>(cs)...)) {}

  template <linalg::RnSpace DecVarDom>
  constexpr auto canonicalize_impl(
      const ConeMapBuilder<DecVarDom>& builder) const {
    auto cones = std::make_tuple();
    auto canonicalize_child = [&](const auto& child) {
      const auto canon = child.canonicalize(builder);
      /// (dakre) TODO update linear selector
      const auto child_sel = std::get<0>(canon);

      /// (dakre) TODO concatenate cones to tuple
      const auto child_cb = std::get<1>(canon);
      cones = std::tuple_cat(cones, child_cb.cones);
    };
    std::apply([&](const auto&... child) { (canonicalize_child(child), ...); },
               this->children);
  }

  /// TODO (dakre) this is more of a debug util however i need to ensure
  /// dom-codom and sum is correct for canonicalization... if theres a nice way
  /// to present this for debug then cool if not then ditch it
#if 0
  constexpr std::optional<T> eval(const PointT& x) const {
    T res(0);
    bool invalid(false);
    auto eval_child = [&](const auto& child) {
      const auto rv = child.eval(x);
      if (!rv.has_value())
        invalid = true;
      else
        res += rv.value();
    };
    std::apply([&](const auto&... child) { (eval_child(child), ...); },
               this->children);
    return invalid ? std::nullopt : res;
  }
#endif
};

template <typename Meta, typename... Cs>
struct Intersection : Atom<Meta, Intersection<Meta, Cs...>, Cs...> {
  using BaseT = Atom<Meta, Intersection<Meta, Cs...>, Cs...>;
  using typename BaseT::Domain;
  using typename BaseT::PointT;

  constexpr explicit Intersection(const Cs... cs)
      : BaseT(std::make_tuple(std::forward<Cs>(cs)...)) {}

  constexpr bool contains(const PointT& x) const {
    bool rv = true;
    auto eval_child = [&](const auto& child) { rv &= child.contains(x); };
    std::apply([&](const auto&... child) { (eval_child(child), ...); },
               this->children);
    return rv;
  }
};

}  // namespace convex
