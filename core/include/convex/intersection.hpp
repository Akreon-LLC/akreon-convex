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
 * @brief intersect over tuple
 *
 * @param child sets
 * @param builder
 * @param canoncailze fn
 *
 * @return new builder
 */
template <size_t idx = 0, typename Children, typename Builder, typename Fn>
constexpr auto tuple_fold_cap(Children&& children, Builder&& builder, Fn&& fn) {
  constexpr size_t N = std::tuple_size_v<std::remove_reference_t<Children>>;
  if constexpr (idx == N) return std::forward<Builder>(builder);
  else {
    auto&& child = std::get<idx>(std::forward<Children>(children));
    auto new_builder =
        std::forward<Fn>(fn)(child, std::forward<Builder>(builder));
    return tuple_fold_cap<idx + 1>(std::forward<Children>(children),
                                   std::move(new_builder),
                                   std::forward<Fn>(fn));
  }
}

/**
 * @brief Intersection
 *
 * Set intersection convex op
 */
template <typename Meta, typename... Cs>
struct Intersection : Atom<Meta, Intersection<Meta, Cs...>, Cs...> {
  /// @brief alias to extract base type
  using BaseT = Atom<Meta, Intersection<Meta, Cs...>, Cs...>;
  using typename BaseT::Domain;
  using typename BaseT::PointT;

  /**
   * @brief CTOR
   *
   * @param children parameter pack
   */
  constexpr explicit Intersection(const Cs... cs)
      : BaseT(std::make_tuple(cs...)) {}

  /**
   * @brief CTOR
   *
   * @param tuple of children
   */
  constexpr explicit Intersection(const std::tuple<Cs...>& cs) : BaseT(cs) {}

  /**
   * @brief canonicalize impl
   *
   * @param builder
   *
   * @return new builder
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize_impl(const Builder& builder) const {
    static_assert(Type == AtomType::SET || Type == AtomType::CAP,
                  "Intersection is only supported on convex sets");
    auto canonicalize_child = [&](const auto& child, const auto& builder_in) {
      return child.template canonicalize<AtomType::CAP>(builder_in);
    };
    return tuple_fold_cap(this->children, builder, canonicalize_child);
  }

  /**
   * @brief contains
   *
   * @param point x in intersection
   *
   * @return T/F
   */
  constexpr bool contains(const PointT& x) const {
    bool rv = true;
    auto eval_child = [&](const auto& child) { rv &= child.contains(x); };
    std::apply([&](const auto&... child) { (eval_child(child), ...); },
               this->children);
    return rv;
  }
};

template <typename C0, typename... Cs>
Intersection(const C0&, const Cs&...) -> Intersection<
    IntersectionMeta<dom_space_t<meta_codom_t<atom_meta_t<C0>>>>,
    atom_decay_t<C0>, atom_decay_t<Cs>...>;

} // namespace convex
