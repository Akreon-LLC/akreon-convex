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
#include <cstdint>
#include <cstdio>

namespace convex {

/// @brief forward declaration of pd result
template <linalg::VectorLike Var, linalg::VectorLike ConeVar>
struct PrimalDualResult;

/**
 * @brief IterState
 */
template <linalg::FieldLike T>
struct IterState {
  T mu{};       ///< normalized duality gap
  T mu_aff{};   ///< duality gap of affine-scaling step
  T sigma{};    ///< mehrotra centering correction
  T mu_tgt{};   ///< mehrotra mu target for centering
  T alpha_x{};  ///< max step to dK size
  T alpha_y{};  ///< max step to dK^* size
  T rd_norm{};  ///< dual norm residual size
  T rp_norm{};  ///< primal norm residual size
  T pz_norm{};  ///< newton step in z \in R^n size
  T px_norm{};  ///< newton step in x \in K size
  T py_norm{};  /// <newton step in y \in K^* size

  /**
   * @flags
   *
   * first bit : primal feasible
   * second bit : dual feasible
   * third bit : dual gap minimized
   */
  uint8_t flags{};  ///< status of iter
};

/**
 * @brief iter state to c string
 *
 * @param iter state,
 * @param char buffer
 * @param buffer size
 *
 * @return write position
 */
template <linalg::FieldLike T>
inline size_t iter_state_to_cstr(const IterState<T>& state, char* out,
                                 const size_t out_size) {
  AK_ASSERT(out != nullptr && "iter_state_to_cstr requires non-null output");
  if (out == nullptr || out_size == 0) return 0;

  const int bytes = std::snprintf(
      out, out_size,
      "mu=%g mu_aff=%g sigma=%g mu_tgt=%g ax=%g ay=%g rd=%g rp=%g pz=%g px=%g "
      "py=%g flags=0x%02X",
      static_cast<double>(state.mu), static_cast<double>(state.mu_aff),
      static_cast<double>(state.sigma), static_cast<double>(state.mu_tgt),
      static_cast<double>(state.alpha_x), static_cast<double>(state.alpha_y),
      static_cast<double>(state.rd_norm), static_cast<double>(state.rp_norm),
      static_cast<double>(state.pz_norm), static_cast<double>(state.px_norm),
      static_cast<double>(state.py_norm), state.flags);

  if (bytes < 0) {
    out[0] = '\0';
    return 0;
  }

  if (static_cast<size_t>(bytes) >= out_size) {
    out[out_size - 1] = '\0';
    return out_size - 1;
  }

  return static_cast<size_t>(bytes);
}

/**
 * @brief trace to cstring
 *
 * @param number of iters
 * @param trace (array of iter states)
 * @param buffer char string
 * @param buffer size
 *
 * @return write position (in case buffer to small)
 */
template <linalg::FieldLike T, size_t N>
inline size_t trace_to_cstr(const size_t& num_iter,
                            const std::array<IterState<T>, N>& trace, char* out,
                            const size_t& out_size) {
  AK_ASSERT(out != nullptr && "trace_to_cstr requires non-null output");
  if (out == nullptr || out_size == 0) return 0;

  const size_t used_iters = (num_iter < N) ? num_iter : N;
  size_t write_pos{0};
  for (size_t i = 0; i < used_iters; ++i) {
    const int header =
        std::snprintf(out + write_pos, out_size - write_pos, "[%zu] ", i);
    if (header < 0) break;
    if (static_cast<size_t>(header) >= out_size - write_pos) {
      out[out_size - 1] = '\0';
      return out_size - 1;
    }
    write_pos += static_cast<size_t>(header);

    const size_t body =
        iter_state_to_cstr(trace[i], out + write_pos, out_size - write_pos);
    write_pos += body;
    if (write_pos >= out_size - 1) {
      out[out_size - 1] = '\0';
      return out_size - 1;
    }

    out[write_pos++] = '\n';
    out[write_pos] = '\0';
  }
  return write_pos;
}

/**
 * @brief vector to cstring
 *
 * @param vector
 * @param buffer char string
 * @param buffer size
 *
 * @return write position (in case buffer to small)
 */
template <linalg::VectorLike Vec>
inline size_t vec_to_cstr(const Vec& v, char* out, const size_t& out_size) {
  AK_ASSERT(out != nullptr && "vec_to_cstr requires non-null output");
  if (out == nullptr || out_size == 0) return 0;

  size_t write_pos{0};
  const int open = std::snprintf(out + write_pos, out_size - write_pos, "[");
  if (open < 0) {
    out[0] = '\0';
    return 0;
  }
  if (static_cast<size_t>(open) >= out_size - write_pos) {
    out[out_size - 1] = '\0';
    return out_size - 1;
  }
  write_pos += static_cast<size_t>(open);

  constexpr size_t N = linalg::vec_size_v<Vec>;
  for (size_t i = 0; i < N; ++i) {
    const char* fmt = (i + 1 == N) ? "%g" : "%g, ";
    const int bytes = std::snprintf(out + write_pos, out_size - write_pos, fmt,
                                    static_cast<double>(v(i)));
    if (bytes < 0) {
      out[0] = '\0';
      return 0;
    }
    if (static_cast<size_t>(bytes) >= out_size - write_pos) {
      out[out_size - 1] = '\0';
      return out_size - 1;
    }
    write_pos += static_cast<size_t>(bytes);
  }

  const int close = std::snprintf(out + write_pos, out_size - write_pos, "]");
  if (close < 0) {
    out[0] = '\0';
    return 0;
  }
  if (static_cast<size_t>(close) >= out_size - write_pos) {
    out[out_size - 1] = '\0';
    return out_size - 1;
  }
  write_pos += static_cast<size_t>(close);
  return write_pos;
}

/**
 * @brief primal-dual result to cstring
 *
 * @param result
 * @param buffer char string
 * @param buffer size
 *
 * @return write position (in case buffer to small)
 */
template <linalg::VectorLike Var, linalg::VectorLike ConeVar>
inline size_t result_to_cstr(const PrimalDualResult<Var, ConeVar>& res,
                             char* out, const size_t& out_size) {
  AK_ASSERT(out != nullptr && "result_to_cstr requires non-null output");
  if (out == nullptr || out_size == 0) return 0;

  size_t write_pos{0};
  const auto append_text = [&](const char* text) -> bool {
    const int bytes =
        std::snprintf(out + write_pos, out_size - write_pos, "%s", text);
    if (bytes < 0) {
      out[0] = '\0';
      return false;
    }
    if (static_cast<size_t>(bytes) >= out_size - write_pos) {
      out[out_size - 1] = '\0';
      write_pos = out_size - 1;
      return false;
    }
    write_pos += static_cast<size_t>(bytes);
    return true;
  };

  const auto rd_norm = linalg::L2<decltype(res.rd)>()(res.rd);
  const auto rp_norm = linalg::L2<decltype(res.rp)>()(res.rp);

  const int header = std::snprintf(
      out + write_pos, out_size - write_pos,
      "tag=%u num_iter=%zu mu=%g convex_obj=%g dual_obj=%g rd=%g rp=%g\n",
      static_cast<unsigned>(res.tag), res.num_iter, static_cast<double>(res.mu),
      static_cast<double>(res.convex_obj), static_cast<double>(res.dual_obj),
      static_cast<double>(rd_norm), static_cast<double>(rp_norm));
  if (header < 0) {
    out[0] = '\0';
    return 0;
  }
  if (static_cast<size_t>(header) >= out_size - write_pos) {
    out[out_size - 1] = '\0';
    return out_size - 1;
  }
  write_pos += static_cast<size_t>(header);

  if (!append_text("z=")) return write_pos;
  write_pos += vec_to_cstr(res.z, out + write_pos, out_size - write_pos);
  if (write_pos >= out_size - 1) return out_size - 1;
  out[write_pos++] = '\n';
  out[write_pos] = '\0';

  if (!append_text("x=")) return write_pos;
  write_pos += vec_to_cstr(res.x, out + write_pos, out_size - write_pos);
  if (write_pos >= out_size - 1) return out_size - 1;
  out[write_pos++] = '\n';
  out[write_pos] = '\0';

  if (!append_text("y=")) return write_pos;
  write_pos += vec_to_cstr(res.y, out + write_pos, out_size - write_pos);
  if (write_pos >= out_size - 1) return out_size - 1;
  out[write_pos++] = '\n';
  out[write_pos] = '\0';

  if (!append_text("rd=")) return write_pos;
  write_pos += vec_to_cstr(res.rd, out + write_pos, out_size - write_pos);
  if (write_pos >= out_size - 1) return out_size - 1;
  out[write_pos++] = '\n';
  out[write_pos] = '\0';

  if (!append_text("rp=")) return write_pos;
  write_pos += vec_to_cstr(res.rp, out + write_pos, out_size - write_pos);
  if (write_pos >= out_size - 1) return out_size - 1;
  out[write_pos++] = '\n';
  out[write_pos] = '\0';

  return write_pos;
}

}  // namespace convex
