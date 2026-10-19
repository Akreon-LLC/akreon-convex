/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdio>

#include "akreon/linalg/assert.hpp"
#include "akreon/linalg/linalg.hpp"

namespace convex::debug {

/**
 * @brief Small no-heap append-only writer for debug formatting.
 *
 * The writer always keeps the output NUL-terminated when capacity is nonzero.
 * `ok()` reports whether every append fit.
 */
struct FixedBufferWriter {
  char* out{};
  std::size_t cap{};
  std::size_t pos{};
  bool truncated{};

  constexpr std::size_t size() const { return pos; }
  constexpr bool ok() const { return !truncated; }
  constexpr const char* c_str() const { return out; }

  void init() {
    if (out != nullptr && cap > 0) out[0] = '\0';
  }

  bool append(const char* text) {
    return format("%s", text == nullptr ? "" : text);
  }

  template <typename... Args>
  bool format(const char* fmt, Args... args) {
    AK_ASSERT(out != nullptr && "FixedBufferWriter requires non-null output");
    if (out == nullptr || cap == 0) return false;
    if (pos >= cap) {
      out[cap - 1] = '\0';
      truncated = true;
      return false;
    }

    const int bytes = std::snprintf(out + pos, cap - pos, fmt, args...);
    if (bytes < 0) {
      out[0] = '\0';
      pos = 0;
      truncated = true;
      return false;
    }

    const auto n = static_cast<std::size_t>(bytes);
    if (n >= cap - pos) {
      pos = cap - 1;
      out[pos] = '\0';
      truncated = true;
      return false;
    }

    pos += n;
    return true;
  }
};

template <std::size_t N>
inline FixedBufferWriter make_writer(char (&out)[N]) {
  FixedBufferWriter w{out, N, 0, false};
  w.init();
  return w;
}

template <linalg::Field T, std::size_t N>
inline bool append_array_slice(FixedBufferWriter& w, const std::array<T, N>& v,
                               std::size_t size) {
  if (!w.append("[")) return false;
  const std::size_t used = (size < N) ? size : N;
  for (std::size_t i = 0; i < used; ++i) {
    if (!w.format((i + 1 == used) ? "%g" : "%g, ", static_cast<double>(v[i])))
      return false;
  }
  return w.append("]");
}

template <linalg::VectorContainer Vec>
inline bool append_vector(FixedBufferWriter& w, const Vec& v) {
  if (!w.append("[")) return false;
  constexpr std::size_t N = linalg::vector_dim_v<Vec>;
  for (std::size_t i = 0; i < N; ++i) {
    if (!w.format((i + 1 == N) ? "%g" : "%g, ", static_cast<double>(v(i))))
      return false;
  }
  return w.append("]");
}

template <linalg::MatrixContainer Mat>
inline bool append_matrix(FixedBufferWriter& w, const Mat& M) {
  if (!w.append("[")) return false;
  constexpr std::size_t Rows = linalg::mat_rows_v<Mat>;
  constexpr std::size_t Cols = linalg::mat_cols_v<Mat>;
  for (std::size_t i = 0; i < Rows; ++i) {
    if (!w.append("[")) return false;
    for (std::size_t j = 0; j < Cols; ++j) {
      if (!w.format((j + 1 == Cols) ? "%g" : "%g, ",
                    static_cast<double>(M(i, j))))
        return false;
    }
    if (!w.append((i + 1 == Rows) ? "]" : "],")) return false;
  }
  return w.append("]");
}

template <linalg::VectorContainer Vec>
inline std::size_t vector_to_cstr(const Vec& v, char* out,
                                  std::size_t out_size) {
  FixedBufferWriter w{out, out_size, 0, false};
  w.init();
  append_vector(w, v);
  return w.size();
}

template <linalg::MatrixContainer Mat>
inline std::size_t matrix_to_cstr(const Mat& M, char* out,
                                  std::size_t out_size) {
  FixedBufferWriter w{out, out_size, 0, false};
  w.init();
  append_matrix(w, M);
  return w.size();
}

} // namespace convex::debug
