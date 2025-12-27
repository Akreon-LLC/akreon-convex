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

namespace convex {

template <typename Meta, typename Derived, typename... Cs>
struct Atom {
  using Domain = meta_dom_t<Meta>;
  using Codomain = meta_codom_t<Meta>;
  using T = dom_type_t<Domain>;
  using PointT = dom_point_t<Domain>;

  static constexpr size_t N = dom_size_v<Domain>;
  static constexpr CurvatureType Curvature = meta_curv_v<Meta>;
  static constexpr SignType DomSign = dom_sign_v<Domain>;
  static constexpr SignType CodomSign = dom_sign_v<Codomain>;

  using LinearMatT = linalg::Matrix<T, N, N>;
  using TransT = linalg::Vector<T, N>;

  constexpr explicit Atom() {}

  constexpr explicit Atom(const std::tuple<Cs...> child_in)
      : children(std::move(child_in)) {}

  constexpr explicit Atom(const affine::AffineMap<LinearMatT>& map_in)
      : map(map_in) {}

  constexpr explicit Atom(const std::tuple<Cs...> child_in,
                          const affine::AffineMap<LinearMatT>& map_in)
      : children(std::move(child_in)), map(map_in) {}

  constexpr auto mul(const LinearMatT& alpha) const {
    return self().mul_impl(alpha);
  }

  constexpr auto add(const TransT& beta) const { return self().add_impl(beta); }

  constexpr auto mul(const T& alpha) const { return self().mul_impl(alpha); }

  constexpr auto add(const T& beta) const { return self().add_impl(beta); }

  constexpr auto canonicalize() const { return self().canonicalize_impl(); }

  constexpr const Derived& self() const {
    return static_cast<const Derived&>(*this);
  }

  std::tuple<Cs...> children;
  affine::AffineMap<LinearMatT> map;
};

}  // namespace convex
