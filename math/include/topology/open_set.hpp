/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <span>

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

constexpr auto constexpr_abs(const auto& x) { return x < 0 ? -1 * x : x; }

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

  constexpr bool operator==(const OpenSet& other) const {
    if (sz != other.sz) return false;
    for (size_t i = 0; i < sz; ++i)
      if (data[i] != other.data[i]) return false;
    return true;
  }

  constexpr bool operator!=(const OpenSet& other) const {
    if (sz != other.sz) return true;
    for (size_t i = 0; i < sz; ++i)
      if (data[i] != other.data[i]) return true;
    return false;
  }

  constexpr T operator()(const size_t idx) const { return data[idx]; }

  constexpr bool is_open(const auto& r) const {
    bool rv = true;
    if constexpr (!is_iterable_v<T>) {
      for (size_t i = 0; i < sz; ++i) rv &= constexpr_abs(data[i]) < r;
    } else {
      for (const auto& y : data) {
        const auto size = y.size();
        for (size_t i = 0; i < size; ++i) rv &= constexpr_abs(y(i)) < r;
      }
    }
    return rv;
  }

  constexpr auto begin() const { return data.begin(); }

  constexpr auto end() const { return data.end(); }

  constexpr bool count(const T& x) const { return find(x).has_value(); }

  constexpr bool insert(const T& v) {
    if (!count(v) && sz < N) {
      data[sz++] = v;
      return true;
    }
    return false;
  }

  constexpr bool insert(std::array<T, N>::const_iterator start,
                        std::array<T, N>::const_iterator end) {
    for (auto it = start; it != end; ++it)
      if (!insert(*it)) return false;
    return true;
  }

  constexpr bool insert(const std::initializer_list<T>& init) {
    for (const auto& x : init)
      if (!insert(x)) return false;
    return true;
  }

  constexpr size_t size() const { return sz; }

  constexpr bool remove(const T& x) {
    const auto rv = find(x);
    if (rv.has_value()) {
      for (size_t i = rv + 1; i < data.size(); ++i)
        std::swap(data[i], data[i - 1]);
      clear_element(data[sz--]);
      return true;
    }
    return false;
  }

  constexpr void clear() {
    for (auto& x : data) clear_element(x);
    sz = 0;
  }

  constexpr auto view() { return std::span(data.data(), sz); }

 private:
  constexpr void clear_element(auto& x) {
    x = std::is_arithmetic_v<T> ? std::numeric_limits<T>::max() : T{};
  }

  constexpr std::optional<size_t> find(const T& x) const {
    if constexpr (!is_iterable_v<T>) {
      for (size_t i = 0; i < data.size(); ++i)
        if (data[i] == x) return i;
    } else {
      const size_t x_sz = std::distance(std::begin(x), std::end(x));
      for (size_t i = 0; i < data.size(); ++i) {
        size_t cnt = 0;
        for (const auto& elem : data[i]) {
          for (const auto& val : x) {
            if (val == elem) {
              cnt++;
              break;
            }
          }
        }
        if (cnt == x_sz) return i;
      }
    }
    return std::nullopt;
  }

  std::array<T, N> data;
  size_t sz;
};

// Deduction guide
template <typename... Ts>
OpenSet(Ts...) -> OpenSet<std::common_type_t<Ts...>, sizeof...(Ts)>;

template <typename T, size_t N1, size_t N2, size_t K = N1 + N2>
constexpr auto set_union(const OpenSet<T, N1>& A, const OpenSet<T, N2>& B) {
  OpenSet<T, K> C;
  for (const auto& x : A) C.insert(x);
  for (const auto& x : B) C.insert(x);
  return C;
}

template <typename T, size_t N1, size_t N2>
constexpr auto operator|(const OpenSet<T, N1>& A, const OpenSet<T, N2>& B) {
  return set_union(A, B);
}

template <typename T, size_t N1, size_t N2, size_t K = N1>
constexpr auto set_intersection(const OpenSet<T, N1>& A,
                                const OpenSet<T, N2>& B) {
  OpenSet<T, K> C;
  if (N1 < N2) {
    for (const auto& x : A)
      if (B.count(x)) C.insert(x);
  } else {
    for (const auto& x : B)
      if (A.count(x)) C.insert(x);
  }
  return C;
}

template <typename T, size_t N1, size_t N2>
constexpr auto operator&(const OpenSet<T, N1>& A, const OpenSet<T, N2>& B) {
  return set_intersection(A, B);
}

template <typename T, size_t N1, size_t N2, size_t K = N1 + N2>
constexpr auto set_difference(const OpenSet<T, N1>& A,
                              const OpenSet<T, N2>& B) {
  const auto C = set_union(A, B);
  const auto D = set_intersection(A, B);
  OpenSet<T, K> result;
  for (const auto& x : C)
    if (!D.count(x)) result.insert(x);

  return result;
}

template <typename T, size_t N1, size_t N2>
constexpr auto operator-(const OpenSet<T, N1>& A, const OpenSet<T, N2>& B) {
  return set_difference(A, B);
}

template <typename T, size_t N1, size_t N2, size_t K = N1 + N2>
constexpr auto set_complement(const OpenSet<T, N1>& universe,
                              const OpenSet<T, N2>& U) {
  return set_difference(universe, U);
}

}  // namespace topology
