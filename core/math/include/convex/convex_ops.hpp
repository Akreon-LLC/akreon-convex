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

template <typename Meta, typename... Cs>
struct Sum : Atom<Meta, Sum<Meta, Cs...>, Cs...> {
  using BaseT = Atom<Meta, Sum<Meta, Cs...>, Cs...>;
  using typename BaseT::Domain;
  static_assert(dom_size_v<Domain> == sizeof...(Cs));

  constexpr explicit Sum(const Cs... cs)
      : BaseT(std::make_tuple(std::forward<Cs>(cs)...)) {}
};

}  // namespace convex
