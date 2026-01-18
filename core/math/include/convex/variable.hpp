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

#include "algebra/solver_traits.hpp"
#include "convex_traits.hpp"
#include "convex_util.hpp"

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
  using BaseT = Atom<Meta, ScalarView<Meta, Child>, Child>;
  using typename BaseT::Domain;
  using typename BaseT::T;

  static constexpr size_t N = dom_size_v<meta_dom_t<atom_meta_t<Child>>>;

  using SelMatT = linalg::Matrix<T, 1, N>;
  using TransT = linalg::Vector<T, 1>;

  constexpr explicit ScalarView(const size_t& idx_in, const Child& child_in)
      : BaseT(std::make_tuple(child_in)),
        map(affine::make_selector_map<SelMatT>(idx_in)),
        idx(idx_in) {
    if (!is_var_type_v<Child>)
      throw std::domain_error(
          "Scalar view child must be the decision variable");
  }

  constexpr explicit ScalarView(const size_t& idx_in, const Child& child_in,
                                const affine::AffineMap<SelMatT>& map_in)
      : BaseT(std::make_tuple(child_in)), map(map_in), idx(idx_in) {
    if (!is_var_type_v<Child>)
      throw std::domain_error(
          "Scalar view child must be the decision variable");
  }

  constexpr auto mul(const T& alpha) const {
    affine::AffineMap<SelMatT> new_map(alpha * map.linear_part(),
                                       map.translation());
    return ScalarView(idx, std::get<0>(this->children), new_map);
  }

  constexpr auto add(const T& beta) const {
    affine::AffineMap<SelMatT> new_map(map.linear_part(),
                                       TransT{beta} + map.translation());
    return ScalarView(idx, std::get<0>(this->children), new_map);
  }

  template <AtomType Type, BuilderLike Builder>
  constexpr auto canonicalize_impl(const Builder& builder) const {
    const auto cb = builder.cb;
    return ConeBlockBuilder<builder_dom_t<Builder>, decltype(map),
                            decltype(cb)>(map, cb);
  }

  affine::AffineMap<SelMatT> map;
  size_t idx;
};

/// @brief variable definition
template <typename Meta>
struct Variable : Atom<Meta, Variable<Meta>> {
  using BaseT = Atom<Meta, Variable<Meta>>;
  using BaseT::DomSign;
  using BaseT::N;
  using typename BaseT::Domain;
  using typename BaseT::T;

  using LinearMatT = linalg::Matrix<T, N, N>;
  using TransT = linalg::Vector<T, N>;

  constexpr explicit Variable()
      : BaseT(), map(affine::make_map_id<LinearMatT>()) {}

  constexpr explicit Variable(const affine::AffineMap<LinearMatT>& map_in)
      : BaseT(), map(map_in) {}

  using ScalarDom = ConvexDomain<linalg::Matrix<T, 1, 1>, DomSign>;
  using ScalarMeta =
      ConvexAtomMeta<ScalarDom, ScalarDom, CurvatureType::AFFINE>;
  using ScalarViewT = ScalarView<ScalarMeta, Variable<Meta>>;
  constexpr std::optional<ScalarViewT> operator()(const size_t& idx) const {
    if (idx >= N) return std::nullopt;
    affine::AffineMap<linalg::Matrix<T, 1, N>> scalar_map(
        this->map.linear_part()(idx), this->map.translation()(idx));
    return ScalarViewT(idx, *this, scalar_map);
  }

  constexpr auto mul(const T& alpha) const {
    affine::AffineMap<LinearMatT> new_map(alpha * map.linear_part(),
                                          map.translation());
    return Variable(new_map);
  }

  constexpr auto mul(const LinearMatT& alpha) const {
    affine::AffineMap<LinearMatT> new_map(alpha * map.linear_part(),
                                          map.translation());
    return Variable(new_map);
  }

  constexpr auto add(const TransT& beta) const {
    affine::AffineMap<LinearMatT> new_map(map.linear_part(),
                                          TransT{beta} + map.translation());
    return Variable(new_map);
  }

  template <AtomType Type, BuilderLike Builder>
  constexpr auto canonicalize_impl(const Builder& builder) const {
    return ConeBlockBuilder<builder_dom_t<Builder>, decltype(map),
                            decltype(builder.cb)>(map, builder.cb);
  }

  template <AtomType Type, BuilderLike Builder>
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
        const auto eps = linalg::SolverTraits<T>::tol;
        for (size_t i = 0; i < N; ++i) g[i] = T(-1) * eps;
      } else if constexpr (SignType::NEG == sign || SignType::ND == sign) {
        F = T(-1) * F;
        const auto eps = linalg::SolverTraits<T>::tol;
        for (size_t i = 0; i < N; ++i) g[i] = T(-1) * eps;
      } else {
        throw std::domain_error("Unknown sign in variable");
      }
      const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);

      const auto new_builder =
          ConeBlockBuilder<builder_dom_t<decltype(builder)>,
                           decltype(coord_map_up), decltype(child_cb)>(
              coord_map_up, child_cb);
      if constexpr (!is_sdp) {
        const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, N>>();
        const auto cones = std::make_tuple(rplus);
        return new_builder.template make<Type>(cones);
      } else {
        const auto psd_cone =
            conic::PositiveSemidefiniteCone<linalg::Vector<T, N>>();
        const auto cones = std::make_tuple(psd_cone);
        return new_builder.template make<Type>(cones);
      }
    }
  }

  /** @brief member vars */
  affine::AffineMap<LinearMatT> map;
};

}  // namespace convex
