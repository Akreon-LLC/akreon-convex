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
#include <tuple>

#include "cone_block_builder.hpp"
#include "config/assert.hpp"
#include "config/solver_traits.hpp"
#include "convex_traits.hpp"

namespace convex {

/// @brief alias for free var domain
template <linalg::RnSpace Dom>
using FreeVarDom = ConvexDomain<Dom, SignType::FREE>;

/// @brief alias for free var meta
template <linalg::RnSpace Dom>
using FreeVarMeta =
    ConvexAtomMeta<FreeVarDom<Dom>, FreeVarDom<Dom>, CurvatureType::AFFINE>;

/// @brief alias for nonneg var domain
template <linalg::RnSpace Dom>
using NonnegVarDom = ConvexDomain<Dom, SignType::NONNEG>;

/// @brief alias for nonneg var meta
template <linalg::RnSpace Dom>
using NonnegVarMeta =
    ConvexAtomMeta<NonnegVarDom<Dom>, NonnegVarDom<Dom>, CurvatureType::AFFINE>;

/// @brief alias for var domain
template <linalg::RnSpace Dom>
using PosVarDom = ConvexDomain<Dom, SignType::POS>;

/// @brief alias for pos var meta
template <linalg::RnSpace Dom>
using PosVarMeta =
    ConvexAtomMeta<PosVarDom<Dom>, PosVarDom<Dom>, CurvatureType::AFFINE>;

/// @brief alias for nonpos domain
template <linalg::RnSpace Dom>
using NonposVarDom = ConvexDomain<Dom, SignType::NONPOS>;

/// @brief alias for nonpos meta
template <linalg::RnSpace Dom>
using NonposVarMeta =
    ConvexAtomMeta<NonposVarDom<Dom>, NonposVarDom<Dom>, CurvatureType::AFFINE>;

/// @brief alias for neg domain
template <linalg::RnSpace Dom>
using NegVarDom = ConvexDomain<Dom, SignType::NEG>;

/// @brief alias for neg meta
template <linalg::RnSpace Dom>
using NegVarMeta =
    ConvexAtomMeta<NegVarDom<Dom>, NegVarDom<Dom>, CurvatureType::AFFINE>;

/// @brief alias for psd var domain
template <linalg::RnSpace Dom>
using PSDVarDom = ConvexDomain<Dom, SignType::NONNEG>;

/// @brief alias for psd var meta
template <linalg::RnSpace Dom>
using PSDVarMeta =
    ConvexAtomMeta<PSDVarDom<Dom>, PSDVarDom<Dom>, CurvatureType::AFFINE>;

/// @brief alias for pd var domain
template <linalg::RnSpace Dom>
using PDVarDom = ConvexDomain<Dom, SignType::NONNEG>;

/// @brief alias for pd var meta
template <linalg::RnSpace Dom>
using PDVarMeta =
    ConvexAtomMeta<PDVarDom<Dom>, PDVarDom<Dom>, CurvatureType::AFFINE>;

/// @brief alias for nsd var domain
template <linalg::RnSpace Dom>
using NSDVarDom = ConvexDomain<Dom, SignType::NONNEG>;

/// @brief alias for nsd var meta
template <linalg::RnSpace Dom>
using NSDVarMeta =
    ConvexAtomMeta<NSDVarDom<Dom>, NSDVarDom<Dom>, CurvatureType::AFFINE>;

/// @brief alias for pd var domain
template <linalg::RnSpace Dom>
using NDVarDom = ConvexDomain<Dom, SignType::NONNEG>;

/// @brief alias for pd var meta
template <linalg::RnSpace Dom>
using NDVarMeta =
    ConvexAtomMeta<NDVarDom<Dom>, NDVarDom<Dom>, CurvatureType::AFFINE>;

template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct ScalarView : Atom<Meta, ScalarView<Meta, Child>, Child> {
  /// @brief extract aliases from types above useful for this class
  using BaseT = Atom<Meta, ScalarView<Meta, Child>, Child>;
  using typename BaseT::Domain;
  using typename BaseT::T;

  static constexpr size_t N = dom_size_v<meta_dom_t<atom_meta_t<Child>>>;

  using SelMatT = linalg::Matrix<T, 1, N>;
  using TransT = linalg::Vector<T, 1>;

  /**
   * @brief CTOR
   *
   * @param index
   * @param child (var)
   */
  constexpr explicit ScalarView(const size_t& idx_in, const Child& child_in)
      : BaseT(std::make_tuple(child_in)),
        map(affine::make_selector_map<SelMatT>(idx_in)),
        idx(idx_in) {
    AK_ASSERT(is_var_type_v<Child> &&
              "ScalarView child must be the decision variable type");
  }

  /**
   * @brief CTOR
   *
   * @param index
   * @param child (var)
   * @param selector map (linear map onto index)
   */
  constexpr explicit ScalarView(const size_t& idx_in, const Child& child_in,
                                const affine::AffineMap<SelMatT>& map_in)
      : BaseT(std::make_tuple(child_in)), map(map_in), idx(idx_in) {
    AK_ASSERT(is_var_type_v<Child> &&
              "ScalarView child must be the decision variable type");
  }

  /**
   * @brief mul (scalar)
   *
   * @param alpha (scalar that scales var)
   *
   * @return new var (updated map)
   */
  constexpr auto mul(const T& alpha) const {
    affine::AffineMap<SelMatT> new_map(alpha * map.linear_part(),
                                       map.translation());
    return ScalarView(idx, std::get<0>(this->children), new_map);
  }

  /**
   * @brief add (scalar)
   *
   * @param beta (scalar)
   *
   * @return new var (updated map)
   */
  constexpr auto add(const T& beta) const {
    affine::AffineMap<SelMatT> new_map(map.linear_part(),
                                       TransT{beta} + map.translation());
    return ScalarView(idx, std::get<0>(this->children), new_map);
  }

  /**
   * @brief canonicalize
   *
   * @param builder (cone block builder)
   *
   * @return new cone block builder
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize_impl(const Builder& builder) const {
    const auto cb = builder.cb;
    return conic::ConeBlockBuilder<conic::builder_dom_t<Builder>, decltype(map),
                                   decltype(cb)>(map, cb);
  }

  /** @brief member var */
  affine::AffineMap<SelMatT> map;  ///< selector map of scalar view onto var
  size_t idx;                      ///< index of orig var
};

/**
 * @brief variable atom
 *
 * Holds the decision var \in R^n
 */
template <typename Meta>
struct Variable : Atom<Meta, Variable<Meta>> {
  /// @brief extract aliases from types above useful for this class
  using BaseT = Atom<Meta, Variable<Meta>>;
  using BaseT::DomSign;
  using BaseT::N;
  using typename BaseT::Domain;
  using typename BaseT::T;

  using LinearMatT = linalg::Matrix<T, N, N>;
  using TransT = linalg::Vector<T, N>;

  using ScalarDom = ConvexDomain<linalg::Matrix<T, 1, 1>, DomSign>;
  using ScalarMeta =
      ConvexAtomMeta<ScalarDom, ScalarDom, CurvatureType::AFFINE>;
  using ScalarViewT = ScalarView<ScalarMeta, Variable<Meta>>;

  /**
   * @brief CTOR (default)
   */
  constexpr explicit Variable()
      : BaseT(), map(affine::make_map_id<LinearMatT>()) {}

  /**
   * @brief CTOR
   *
   * @param linear map in
   */
  constexpr explicit Variable(const affine::AffineMap<LinearMatT>& map_in)
      : BaseT(), map(map_in) {}

  /**
   * @brief operator overload for index
   *
   * @param index
   *
   * @return ScalarView of var at index
   */
  constexpr std::optional<ScalarViewT> operator()(const size_t& idx) const {
    if (idx >= N) return std::nullopt;
    affine::AffineMap<linalg::Matrix<T, 1, N>> scalar_map(
        this->map.linear_part()(idx), this->map.translation()(idx));
    return ScalarViewT(idx, *this, scalar_map);
  }

  /**
   * @brief mul (scalar)
   *
   * @param alpha (scalar that scales var)
   *
   * @return new var (updated map)
   */
  constexpr auto mul(const T& alpha) const {
    affine::AffineMap<LinearMatT> new_map(alpha * map.linear_part(),
                                          map.translation());
    return Variable(new_map);
  }

  /**
   * @brief mul (scalar)
   *
   * @param alpha (linear mat)
   *
   * @return new var (updated map)
   */
  constexpr auto mul(const LinearMatT& alpha) const {
    affine::AffineMap<LinearMatT> new_map(alpha * map.linear_part(),
                                          map.translation());
    return Variable(new_map);
  }

  /**
   * @brief add (scalar)
   *
   * @param beta (scalar)
   *
   * @return new var (updated map)
   */
  constexpr auto add(const TransT& beta) const {
    affine::AffineMap<LinearMatT> new_map(map.linear_part(),
                                          TransT{beta} + map.translation());
    return Variable(new_map);
  }

  /**
   * @brief canonicalize
   *
   * Almost treated as a no-op as var will always be leaf node but need to pass
   * its map and cone block
   *
   * @param builder (ConeBlockBuilder)
   *
   * @return cone block builder (new)
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize_impl(const Builder& builder) const {
    return conic::ConeBlockBuilder<conic::builder_dom_t<Builder>, decltype(map),
                                   decltype(builder.cb)>(map, builder.cb);
  }

  /**
   * @brief split
   *
   * x -> (x(0), x(1), ...) tuple
   *
   * @return tuple(ScalarViews...)
   */
  constexpr auto split() const {
    auto form_tup = [this]<size_t... Is>(std::index_sequence<Is...>) {
      return std::make_tuple(((*this)(Is).value())...);
    };
    return form_tup(std::make_index_sequence<N>{});
  }

  /**
   * @brief add_var_block
   *
   * Instead of canonicalization adding the variable to the block it is treated
   * specially by this function to manually add it to F
   *
   * @param builder (ConeBlockBuilder)
   *
   * @return new cone block builder
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto add_var_block(const Builder& builder) const {
    static constexpr SignType sign = dom_sign_v<meta_dom_t<Meta>>;
    static constexpr bool is_sdp = SignType::PSD == sign ||
                                   SignType::NSD == sign ||
                                   SignType::PD == sign || SignType::ND == sign;
    const auto child_builder = this->template canonicalize<Type>(builder);
    if constexpr (SignType::FREE == sign) {
      return child_builder;
    } else {
      const auto child_cb = child_builder.cb;

      auto F = linalg::make_identity<T, N>();
      auto g = linalg::Vector<T, N>{};
      if constexpr (SignType::NONNEG == sign || SignType::PSD == sign) {
        /// no-op
      } else if constexpr (SignType::NONPOS == sign || SignType::NSD == sign) {
        F = T(-1) * F;
      } else if constexpr (SignType::POS == sign || SignType::PD == sign) {
        const auto eps = linalg::SolverTraits<T>::TOL;
        for (size_t i = 0; i < N; ++i) g(i) = T(-1) * eps;
      } else if constexpr (SignType::NEG == sign || SignType::ND == sign) {
        F = T(-1) * F;
        const auto eps = linalg::SolverTraits<T>::TOL;
        for (size_t i = 0; i < N; ++i) g(i) = T(-1) * eps;
      } else {
        AK_ASSERT(false && "Unsupported sign type in Variable::add_var_block");
      }
      const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);

      const auto new_builder =
          conic::ConeBlockBuilder<conic::builder_dom_t<decltype(builder)>,
                                  decltype(coord_map_up), decltype(child_cb)>(
              coord_map_up, child_cb);
      if constexpr (!is_sdp) {
        const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, N>>();
        const auto cones = std::make_tuple(rplus);
        return new_builder.template make<Type>(cones);
      } else {
        static constexpr size_t K = N * (N + 1) / 2;
        const auto psd_cone =
            conic::PositiveSemidefiniteCone<linalg::Vector<T, K>, N>();
        const auto cones = std::make_tuple(psd_cone);
        return new_builder.template make<Type>(cones);
      }
    }
  }

  /** @brief member vars */
  affine::AffineMap<LinearMatT> map;
};

}  // namespace convex
