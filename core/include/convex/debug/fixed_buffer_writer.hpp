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

#include <array>
#include <cstddef>
#include <cstdio>

#include "config/assert.hpp"
#include "linalg/linalg.hpp"

namespace convex::debug {

/**
 * @brief Small no-heap append-only writer for debug formatting.
 *
 * The writer always keeps the output NUL-terminated when capacity is nonzero.
 * `ok()` reports whether every append fit.
 */
struct FixedBufferWriter {
  char* out{};
  size_t cap{};
  size_t pos{};
  bool truncated{};

  constexpr size_t size() const { return pos; }
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

    const auto n = static_cast<size_t>(bytes);
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

template <size_t N>
inline FixedBufferWriter make_writer(char (&out)[N]) {
  FixedBufferWriter w{out, N, 0, false};
  w.init();
  return w;
}

template <linalg::FieldLike T, size_t N>
inline bool append_array_slice(FixedBufferWriter& w, const std::array<T, N>& v,
                               size_t size) {
  if (!w.append("[")) return false;
  const size_t used = (size < N) ? size : N;
  for (size_t i = 0; i < used; ++i) {
    if (!w.format((i + 1 == used) ? "%g" : "%g, ", static_cast<double>(v[i])))
      return false;
  }
  return w.append("]");
}

template <linalg::VectorLike Vec>
inline bool append_vector(FixedBufferWriter& w, const Vec& v) {
  if (!w.append("[")) return false;
  constexpr size_t N = linalg::vec_size_v<Vec>;
  for (size_t i = 0; i < N; ++i) {
    if (!w.format((i + 1 == N) ? "%g" : "%g, ", static_cast<double>(v(i))))
      return false;
  }
  return w.append("]");
}

template <linalg::MatrixLike Mat>
inline bool append_matrix(FixedBufferWriter& w, const Mat& M) {
  if (!w.append("[")) return false;
  constexpr size_t Rows = linalg::mat_rows_v<Mat>;
  constexpr size_t Cols = linalg::mat_cols_v<Mat>;
  for (size_t i = 0; i < Rows; ++i) {
    if (!w.append("[")) return false;
    for (size_t j = 0; j < Cols; ++j) {
      if (!w.format((j + 1 == Cols) ? "%g" : "%g, ",
                    static_cast<double>(M(i)(j))))
        return false;
    }
    if (!w.append((i + 1 == Rows) ? "]" : "],")) return false;
  }
  return w.append("]");
}

template <linalg::VectorLike Vec>
inline size_t vector_to_cstr(const Vec& v, char* out, size_t out_size) {
  FixedBufferWriter w{out, out_size, 0, false};
  w.init();
  append_vector(w, v);
  return w.size();
}

template <linalg::MatrixLike Mat>
inline size_t matrix_to_cstr(const Mat& M, char* out, size_t out_size) {
  FixedBufferWriter w{out, out_size, 0, false};
  w.init();
  append_matrix(w, M);
  return w.size();
}

} // namespace convex::debug
