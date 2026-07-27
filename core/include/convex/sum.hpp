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

#include "linalg/affine/affine_map.hpp"
#include "cone_traits.hpp"
#include "convex_traits.hpp"

namespace convex {

/**
 * @brief Validate that a sum child has the same codomain as the parent sum.
 *
 * @tparam SumCodomain expected sum codomain
 * @tparam Child child atom type to validate
 *
 * @return void
 */
template <typename SumCodomain, typename Child>
constexpr void validate_sum_child_codomain() {
  using ChildCodomain = meta_codom_t<atom_meta_t<Child>>;
  static_assert(
      std::same_as<dom_space_t<SumCodomain>, dom_space_t<ChildCodomain>>,
      "Sum child codomain must match the sum codomain space");
}

/**
 * @brief Accumulate the affine selector for a sum atom.
 *
 * @param prev current accumulated selector map
 * @param next selector map emitted by the next canonicalized child
 *
 * @return selector map whose affine rows represent the child sum
 */
template <affine::AffineMapLike PrevMap, affine::AffineMapLike NextMap>
constexpr auto update_selector(const PrevMap& prev, const NextMap& next) {
  using PrevMatT = affine::affine_mat_t<PrevMap>;
  using NextMatT = affine::affine_mat_t<NextMap>;
  using T = linalg::matrix_scalar_t<PrevMatT>;

  static constexpr std::size_t PrevDom = linalg::mat_cols_v<PrevMatT>;
  static constexpr std::size_t PrevCodom = linalg::mat_rows_v<PrevMatT>;

  static constexpr std::size_t NextDom = linalg::mat_cols_v<NextMatT>;
  static constexpr std::size_t NextCodom = linalg::mat_rows_v<NextMatT>;

  /// (dakre) first step is seeding the selector map from first child then
  /// accumulate selector thereafter
  if constexpr (PrevCodom != NextCodom) {
    return affine::AffineMap<linalg::remove_cvref_t<NextMatT>>(
        next.linear_part(), next.translation());
  } else {
    static constexpr std::size_t FullDom =
        PrevDom > NextDom ? PrevDom : NextDom;
    linalg::Matrix<T, PrevCodom, FullDom> F{};
    for (std::size_t i = 0; i < PrevCodom; ++i) {
      for (std::size_t j = 0; j < FullDom; ++j) {
        if (j < PrevDom) F(i, j) += prev.linear_part()(i, j);
        if (j < NextDom) F(i, j) += next.linear_part()(i, j);
      }
    }

    linalg::Vector<T, PrevCodom> g{};
    for (std::size_t i = 0; i < PrevCodom; ++i) {
      g(i) = prev.translation()(i) + next.translation()(i);
    }
    return affine::AffineMap<decltype(F)>(F, g);
  }
}

/**
 * @brief Canonicalize and fold all sum children.
 *
 * @tparam idx child index currently being folded
 * @tparam Children tuple-like child container
 * @tparam Builder canonical-form builder type
 * @tparam Fn callable that canonicalizes a child
 * @tparam Map selector-map type
 *
 * @param children atoms
 * @param builder current canonical-form builder
 * @param fn canonicalization callable
 * @param selector current accumulated selector map
 *
 * @return tuple containing the accumulated selector and builder
 */
template <std::size_t idx = 0, typename Children, typename Builder, typename Fn,
          affine::AffineMapLike Map>
constexpr auto tuple_fold_sum(Children&& children, Builder&& builder, Fn&& fn,
                              Map&& selector) {
  constexpr std::size_t N =
      std::tuple_size_v<std::remove_reference_t<Children>>;
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

/**
 * @brief DCP sum atom over two or more child atoms.
 */
template <typename Meta, typename... Cs>
struct Sum : Atom<Meta, Sum<Meta, Cs...>, Cs...> {
  /// @brief CRTP base type.
  using BaseT = Atom<Meta, Sum<Meta, Cs...>, Cs...>;
  /// @brief Parent atom codomain.
  using typename BaseT::Codomain;
  /// @brief Parent atom domain.
  using typename BaseT::Domain;
  /// @brief Point type accepted by the parent domain.
  using typename BaseT::PointT;
  /// @brief Scalar field used by child atoms.
  using typename BaseT::T;
  static_assert(dom_size_v<Domain> == sizeof...(Cs),
                "Size of domain is not equal to children");
  static_assert(sizeof...(Cs) > 1, "Number of children not greater than 1");

  using MatT = linalg::Matrix<T, 1, 1>; ///< Scalar affine map matrix type.

  /**
   * @brief Construct a sum atom from child atoms.
   *
   * @param cs child atoms to sum
   */
  constexpr explicit Sum(const Cs&... cs)
      : BaseT(std::make_tuple(cs...)), map(affine::make_map_id<MatT>()) {
    (validate_sum_child_codomain<Codomain, Cs>(), ...);
  }

  /**
   * @brief Construct a transformed sum atom from a scalar affine map.
   *
   * @param map_in scalar affine transform applied to the child sum
   * @param cs tuple of child atoms
   */
  constexpr explicit Sum(const affine::AffineMap<MatT>& map_in,
                         const std::tuple<Cs...>& cs)
      : BaseT(cs), map(map_in) {
    (validate_sum_child_codomain<Codomain, Cs>(), ...);
  }

  /**
   * @brief Scale this sum by a compile-time scalar.
   *
   * @tparam Scalar scale factor applied to the selector map
   *
   * @return transformed sum atom
   */
  template <T Scalar>
  constexpr auto mul() const {
    affine::AffineMap<MatT> new_map(Scalar * map.linear_part(),
                                    map.translation());
    return Sum<Meta, Cs...>(new_map, this->children);
  }

  /**
   * @brief Add a scalar offset to this sum.
   *
   * @param beta scalar
   *
   * @return transformed sum atom
   */
  constexpr auto add(const T& beta) const {
    affine::AffineMap<MatT> new_map(
        map.linear_part(), linalg::Vector<T, 1>{beta} + map.translation());
    return Sum<Meta, Cs...>(new_map, this->children);
  }

  /**
   * @brief Canonicalize this sum into a cone builder.
   *
   * @tparam Type requested atom canonicalization target
   * @tparam Builder canonical-form builder type
   *
   * @param builder current canonical-form builder
   *
   * @return updated canonical-form builder
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize_impl(const Builder& builder) const {
    static_assert(Type != AtomType::SET, "Minkowski sum is not yet supported");

    auto canonicalize_child = [&](const auto& child, const auto& builder_in) {
      return child.template canonicalize<AtomType::ADD>(builder_in);
    };
    const auto res = tuple_fold_sum(this->children, builder, canonicalize_child,
                                    builder.coord_map);
    const auto selector = std::get<0>(res);
    const auto F = map.linear_part()(0, 0) * selector.linear_part();
    const auto g =
        map.linear_part()(0, 0) * selector.translation() + map.translation();
    const auto new_selector =
        affine::AffineMap<decltype(selector.linear_part())>(F, g);

    const auto new_builder = std::get<1>(res);
    // Summation changes only the current selector; canonical rows and their
    // trace have already been realized by the folded child builders.
    return conic::CanonicalFormBuilder<
        conic::builder_dom_t<Builder>, decltype(new_selector),
        decltype(new_builder.cb), decltype(new_builder.eq_map),
        decltype(new_builder.canonical_trace)>(new_selector, new_builder.cb,
                                               new_builder.eq_map,
                                               new_builder.canonical_trace);
  }

  affine::AffineMap<MatT>
      map; ///< Scalar affine transform applied after summing children.
};

/**
 * @brief Deduction guide for sum atoms.
 */
template <typename C0, typename... Cs>
Sum(const C0&, const Cs&...)
    -> Sum<SumMeta<linalg::Matrix<dom_type_t<meta_dom_t<atom_meta_t<C0>>>, 1,
                                  1 + sizeof...(Cs)>,
                   dom_space_t<meta_codom_t<atom_meta_t<C0>>>>,
           atom_decay_t<C0>, atom_decay_t<Cs>...>;

} // namespace convex
