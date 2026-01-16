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

template <affine::AffineMapLike PrevMap, affine::AffineMapLike NextMap>
constexpr auto update_selector(const PrevMap& prev, const NextMap& next) {
  using PrevMatT = affine::affine_mat_t<PrevMap>;
  using NextMatT = affine::affine_mat_t<NextMap>;
  using T = linalg::mat_type_t<PrevMatT>;

  static constexpr size_t PrevDom = linalg::mat_cols_v<PrevMatT>;
  static constexpr size_t PrevCodom = linalg::mat_rows_v<PrevMatT>;

  static constexpr size_t NextDom = linalg::mat_cols_v<NextMatT>;
  static constexpr size_t NextCodom = linalg::mat_rows_v<NextMatT>;

  if constexpr (PrevDom == 0 && PrevCodom == 0)
    return next;
  else {
    static_assert(PrevCodom == NextCodom,
                  "Codomains must align between previous and next map");

    linalg::Matrix<T, PrevCodom, NextDom> F;
    for (size_t i = 0; i < PrevCodom; ++i) {
      for (size_t j = 0; j < PrevDom; ++j)
        F[i][j] = prev.linear_part()(i)(j) + next.linear_part()(i)(j);
      for (size_t j = PrevDom; j < NextDom; ++j)
        F[i][j] = next.linear_part()(i)(j);
    }

    return affine::AffineMap<decltype(F)>(F, prev.translation());
  }
}

template <size_t idx = 0, typename Children, typename Builder, typename Fn,
          affine::AffineMapLike Map>
constexpr auto tuple_fold_sum(Children&& children, Builder&& builder, Fn&& fn,
                              Map&& selector) {
  constexpr size_t N = std::tuple_size_v<std::remove_reference_t<Children>>;
  if constexpr (idx == N)
    return std::tuple(selector, std::forward<Builder>(builder));
  else {
    auto&& child = std::get<idx>(std::forward<Children>(children));
    auto new_builder =
        std::forward<Fn>(fn)(child, std::forward<Builder>(builder));
    auto new_selector = update_selector(selector, new_builder.coord_map);
    return tuple_fold_sum<idx + 1>(std::forward<Children>(children),
                                   std::move(new_builder), std::forward<Fn>(fn),
                                   std::move(new_selector));
  }
}

template <typename Meta, typename... Cs>
struct Sum : Atom<Meta, Sum<Meta, Cs...>, Cs...> {
  using BaseT = Atom<Meta, Sum<Meta, Cs...>, Cs...>;
  using typename BaseT::Domain;
  using typename BaseT::PointT;
  using typename BaseT::T;
  static_assert(dom_size_v<Domain> == sizeof...(Cs),
                "Size of domain is not equal to children");
  static_assert(sizeof...(Cs) > 1, "Number of children not greater than 1");

  using MatT = linalg::Matrix<T, 1, 1>;

  constexpr explicit Sum(const Cs... cs)
      : BaseT(std::make_tuple(std::forward<Cs>(cs)...)),
        map(affine::make_map_id<MatT>()) {}

  constexpr explicit Sum(const affine::AffineMap<MatT>& map_in,
                         const std::tuple<Cs...>& cs)
      : BaseT(cs), map(map_in) {}

  template <T Scalar>
  constexpr auto mul() const {
    affine::AffineMap<MatT> new_map(Scalar * map.linear_part(),
                                    map.translation());
    return Sum<Meta, Cs...>(new_map, this->children);
  }

  constexpr auto add(const T& beta) const {
    affine::AffineMap<MatT> new_map(map.linear_part(),
                                    beta + map.translation());
    return Sum<Meta, Cs...>(new_map, this->children);
  }

  template <AtomType Type, BuilderLike Builder>
  constexpr auto canonicalize_impl(const Builder& builder) const {
    static_assert(Type != AtomType::SET, "Minkowski sum is not yet supported");

    auto canonicalize_child = [&](const auto& child, const auto& builder_in) {
      return child.template canonicalize<AtomType::ADD>(builder_in);
    };
    const auto res = tuple_fold_sum(this->children, builder, canonicalize_child,
                                    builder.coord_map);
    const auto selector = std::get<0>(res);
    const auto F = map.linear_part()(0)(0) * selector.linear_part();
    const auto g = map.translation() + selector.translation();
    const auto new_selector =
        affine::AffineMap<decltype(selector.linear_part())>(F, g);

    const auto new_builder = std::get<1>(res);
    return ConeBlockBuilder<builder_dom_t<Builder>, decltype(new_selector),
                            decltype(new_builder.cb)>(new_selector,
                                                      new_builder.cb);
  }

  /** @brief member var */
  affine::AffineMap<MatT> map;
};

template <size_t idx = 0, typename Children, typename Builder, typename Fn>
constexpr auto tuple_fold_cap(Children&& children, Builder&& builder, Fn&& fn) {
  constexpr size_t N = std::tuple_size_v<std::remove_reference_t<Children>>;
  if constexpr (idx == N)
    return std::forward<Builder>(builder);
  else {
    auto&& child = std::get<idx>(std::forward<Children>(children));
    auto new_builder =
        std::forward<Fn>(fn)(child, std::forward<Builder>(builder));
    return tuple_fold_cap<idx + 1>(std::forward<Children>(children),
                                   std::move(new_builder),
                                   std::forward<Fn>(fn));
  }
}

template <typename Meta, typename... Cs>
struct Intersection : Atom<Meta, Intersection<Meta, Cs...>, Cs...> {
  using BaseT = Atom<Meta, Intersection<Meta, Cs...>, Cs...>;
  using typename BaseT::Domain;
  using typename BaseT::PointT;

  constexpr explicit Intersection(const Cs... cs)
      : BaseT(std::make_tuple(std::forward<Cs>(cs)...)) {}

  template <AtomType Type, BuilderLike Builder>
  constexpr auto canonicalize_impl(const Builder& builder) const {
    static_assert(Type == AtomType::SET || Type == AtomType::CAP,
                  "Intersection is only supported on convex sets");
    auto canonicalize_child = [&](const auto& child, const auto& builder_in) {
      return child.template canonicalize<AtomType::CAP>(builder_in);
    };
    return tuple_fold_cap(this->children, builder, canonicalize_child);
  }

  constexpr bool contains(const PointT& x) const {
    bool rv = true;
    auto eval_child = [&](const auto& child) { rv &= child.contains(x); };
    std::apply([&](const auto&... child) { (eval_child(child), ...); },
               this->children);
    return rv;
  }
};

}  // namespace convex
