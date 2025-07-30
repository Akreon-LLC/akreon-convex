/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <array>

namespace topology {

// Base set X
template <typename T, size_t N>
struct ConstexprSet {
  using value_type = T;

  constexpr Set(std::initializer_list<T> init) {
    for (const auto& v : init) insert(v);
  }

  constexpr auto begin() const { return data.begin(); }

  constexpr auto end() const { return data.end(); }

  constexpr bool contains(const T& x) const {
    return std::find(begin(), end(), x) != end();
  }

  constexpr void insert(T&& v) {
    if (contains(v)) return;
    static_assert(idx + 1 < N);
    data[idx++] = v;
  }

  constexpr void insert(const std::array<T, N>::iterator& start,
                        const std::array<T, N>::iterator& end) {
    for (auto it = start; it != end; ++it) insert(*it);
  }

  constexpr size_t size() const { return idx; }

 private:
  std::array<T, N> data;
  size_t idx;
};

// Deduction guide
template <typename... Ts>
ConstexprSet(Ts...) -> ConstexprSet<std::common_type_t<Ts...>, sizeof...(Ts)>;

}  // namespace topology
