/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <array>
#include <iostream>

namespace topology {

enum error_t : int { SET_IS_FULL = 1, ALREADY_EXISTS };

template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())),
                                  decltype(std::end(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
constexpr bool is_iterable_v = is_iterable<T>::value;

// Base set X
template <typename T, size_t N>
struct OpenSet {
  using value_type = T;

  static constexpr size_t s_size = N;

  constexpr OpenSet() { clear(); }

  constexpr OpenSet(const std::initializer_list<T>& init) {
    clear();
    insert(init);
  }

  constexpr bool is_open(auto r) const {
    bool rv = true;
    if constexpr (!is_iterable_v<T>) {
      for (const auto& elem : data) rv &= std::abs(elem) < r;
    } else {
      for (const auto& y : data)
        for (const auto& elem : y) rv &= std::abs(elem) < r;
    }
    return rv;
  }

  constexpr auto begin() const { return data.begin(); }

  constexpr auto end() const { return data.end(); }

  constexpr bool contains(const T& x) const {
    if constexpr (!is_iterable_v<T>) {
      for (const auto& elem : data) {
        if (elem == x) return true;
      }
    } else {
      for (const auto& y : data) {
        size_t cnt = 0;
        for (const auto& elem : y) {
          for (const auto& val : x) {
            if (val == elem) {
              cnt++;
              break;
            }
          }
        }
        if (cnt == x.size()) return true;
      }
    }
    return false;
  }

  constexpr bool count(const T& x) const { return contains(x); }

  constexpr void insert(const T& v) {
    if (contains(v)) throw ALREADY_EXISTS;
    if (sz == N) throw SET_IS_FULL;
    data[sz++] = v;
  }

  constexpr void insert(std::array<T, N>::const_iterator start,
                        std::array<T, N>::const_iterator end) {
    for (auto it = start; it != end; ++it) insert(*it);
  }

  constexpr void insert(const std::initializer_list<T>& init) {
    for (const auto& x : init) insert(x);
  }

  constexpr size_t size() const { return sz; }

  constexpr void clear() {
    for (auto& x : data) x = std::numeric_limits<T>::max();
    sz = 0;
  }

 private:
  std::array<T, N> data;
  size_t sz;
};

// Deduction guide
template <typename... Ts>
OpenSet(Ts...) -> OpenSet<std::common_type_t<Ts...>, sizeof...(Ts)>;

}  // namespace topology
