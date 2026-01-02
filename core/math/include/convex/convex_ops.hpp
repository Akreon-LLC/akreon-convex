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

template <size_t idx = 0, typename Children, typename Builder, typename Fn>
constexpr auto tuple_fold_left(Children&& children, Builder&& builder,
                               Fn&& fn) {
  constexpr size_t N = std::tuple_size_v<std::remove_reference_t<Children>>;
  if constexpr (idx == N)
    return std::forward<Builder>(builder);
  else {
    auto&& child = std::get<idx>(std::forward<Children>(children));
    auto new_builder =
        std::forward<Fn>(fn)(child, std::forward<Builder>(builder));
    return tuple_fold_left<idx + 1>(std::forward<Children>(children),
                                    std::move(new_builder),
                                    std::forward<Fn>(fn));
  }
}

template <typename Meta, typename... Cs>
struct Sum : Atom<Meta, Sum<Meta, Cs...>, Cs...> {
  using BaseT = Atom<Meta, Sum<Meta, Cs...>, Cs...>;
  using typename BaseT::Domain;
  using typename BaseT::PointT;
  using typename BaseT::T;
  static_assert(dom_size_v<Domain> == sizeof...(Cs));

  constexpr explicit Sum(const Cs... cs)
      : BaseT(std::make_tuple(std::forward<Cs>(cs)...)) {}

  template <linalg::RnSpace DecVarDom, affine::AffineMapLike Map,
            conic::ConeBlockLike CB>
  constexpr auto canonicalize_impl(
      const ConeBlockBuilder<DecVarDom, Map, CB>& builder) const {
    auto canonicalize_child = [&](const auto& child, const auto& builder_in) {
      return child.canonicalize(builder_in);
    };
    // (dakre) TODO return reduced cone block with map and cone blocks
    return tuple_fold_left(this->children, builder, canonicalize_child);
  }
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
