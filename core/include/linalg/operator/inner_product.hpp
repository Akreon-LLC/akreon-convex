/* Copyright © 2026 Akreon LLC. All rights reserved.
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

#include <complex>
#include <concepts>
#include <type_traits>

#include "linalg/vector/vector_concepts.hpp"

namespace linalg {

template <FieldVector Vec>
struct StandardInnerProduct {
  using domain_type = remove_cvref_t<Vec>;
  using scalar_type = vector_element_t<domain_type>;
  using codomain_type = scalar_type;

  constexpr codomain_type operator()(const domain_type& lhs,
                                     const domain_type& rhs) const {
    codomain_type result{};
    for (std::size_t i = 0; i < vector_dim_v<domain_type>; ++i) {
      if constexpr (Complex<scalar_type>) result += std::conj(lhs(i)) * rhs(i);
      else result += lhs(i) * rhs(i);
    }
    return result;
  }
};

template <class Product>
struct inner_product_traits;

template <FieldVector Vec>
struct inner_product_traits<StandardInnerProduct<Vec>> {
  using domain_type = remove_cvref_t<Vec>;
  using scalar_type = vector_element_t<domain_type>;
  using codomain_type = scalar_type;
};

template <class Product>
using inner_product_traits_t = inner_product_traits<remove_cvref_t<Product>>;

template <class Product>
using inner_product_domain_t =
    typename inner_product_traits_t<Product>::domain_type;

template <class Product>
using inner_product_scalar_t =
    typename inner_product_traits_t<Product>::scalar_type;

template <class Product>
using inner_product_codomain_t =
    typename inner_product_traits_t<Product>::codomain_type;

template <class Product>
concept InnerProduct = requires(const remove_cvref_t<Product>& product,
                                const inner_product_domain_t<Product>& lhs,
                                const inner_product_domain_t<Product>& rhs) {
  { product(lhs, rhs) } -> std::same_as<inner_product_codomain_t<Product>>;
};

template <FieldVector Vec>
constexpr auto inner_product(const Vec& lhs, const Vec& rhs) {
  return StandardInnerProduct<remove_cvref_t<Vec>>{}(lhs, rhs);
}

} // namespace linalg
