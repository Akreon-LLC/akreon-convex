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

#include "config/solver_config.hpp"

namespace detail {

template <typename Cone>
struct ConeTypeName {
  static constexpr const char* value = "CONE";
};

template <linalg::VectorLike Vec>
struct ConeTypeName<conic::NonnegOrthantCone<Vec>> {
  static constexpr const char* value = "RPLUS";
};

template <linalg::VectorLike Vec>
struct ConeTypeName<conic::SecondOrderCone<Vec>> {
  static constexpr const char* value = "SOC";
};

template <linalg::VectorLike Vec, size_t N>
struct ConeTypeName<conic::PositiveSemidefiniteCone<Vec, N>> {
  static constexpr const char* value = "PSD";
};

template <linalg::VectorLike Vec>
struct ConeTypeName<conic::ExpCone<Vec>> {
  static constexpr const char* value = "EXP";
};

template <linalg::VectorLike Vec>
struct ConeTypeName<conic::PowerCone<Vec>> {
  static constexpr const char* value = "POW";
};

template <typename Cone>
constexpr const char* cone_type_name_v =
    ConeTypeName<linalg::remove_cvref_t<Cone>>::value;

}  // namespace detail

namespace convex {

/// @brief forward declaration of pd result
template <linalg::VectorLike Var, linalg::VectorLike ConeVar>
struct PrimalDualResult;

/**
 * @brief IterState
 */
template <linalg::FieldLike T, size_t MaxConeDim, size_t VarDim>
struct PredictorCorrectorConeTrace {
  T mu{};      ///< local normalized duality gap (per cone)
  T rhs_c{};   ///< ||rhs_c||_2 in local cone block
  T pz_rhs{};  ///< ||pz_rhs||_2 after cone contribution
  std::array<T, MaxConeDim> xk{};
  std::array<T, MaxConeDim> yk{};
  std::array<T, MaxConeDim> rhs_c_vec{};
  std::array<T, MaxConeDim> px{};
  std::array<T, MaxConeDim> py{};
  std::array<T, VarDim> pz{};
  size_t cone_dim{};
  const char* cone_type{"unknown"};
  bool active{false};
};

/**
 * @brief IterState
 */
template <linalg::FieldLike T, size_t NumCones, size_t MaxConeDim, size_t VarDim>
struct IterState {
  T mu{};       ///< normalized duality gap
  T alpha_x{};  ///< max step to dK size
  T alpha_y{};  ///< max step to dK^* size
  T rd_norm{};  ///< dual norm residual size
  T rp_norm{};  ///< primal norm residual size
  T pz_norm{};  ///< newton step in z \in R^n size
  T px_norm{};  ///< newton step in x \in K size
  T py_norm{};  ///< newton step in y \in K^* size
  std::array<PredictorCorrectorConeTrace<T, MaxConeDim, VarDim>, NumCones>
      predictor{};
  std::array<PredictorCorrectorConeTrace<T, MaxConeDim, VarDim>, NumCones>
      corrector{};
  config::SolverResult tag{};  ///< tag of current iterate

  constexpr void reset_pc_trace() {
    for (size_t i = 0; i < NumCones; ++i) {
      predictor[i] = PredictorCorrectorConeTrace<T, MaxConeDim, VarDim>{};
      corrector[i] = PredictorCorrectorConeTrace<T, MaxConeDim, VarDim>{};
    }
  }
};

template <linalg::FieldLike T, size_t N>
inline size_t array_slice_to_cstr(const std::array<T, N>& v, const size_t size,
                                  char* out, const size_t out_size) {
  AK_ASSERT(out != nullptr && "array_slice_to_cstr requires non-null output");
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

  for (size_t i = 0; i < size; ++i) {
    const char* fmt = (i + 1 == size) ? "%g" : "%g, ";
    const int bytes = std::snprintf(out + write_pos, out_size - write_pos, fmt,
                                    static_cast<double>(v[i]));
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
 * @brief iter state to c string
 *
 * @param iter state,
 * @param char buffer
 * @param buffer size
 *
 * @return write position
 */
template <linalg::FieldLike T, size_t NumCones, size_t MaxConeDim, size_t VarDim>
inline size_t iter_state_to_cstr(
    const IterState<T, NumCones, MaxConeDim, VarDim>& state, char* out,
    const size_t out_size) {
  AK_ASSERT(out != nullptr && "iter_state_to_cstr requires non-null output");
  if (out == nullptr || out_size == 0) return 0;

  size_t write_pos{0};
  const auto appendf = [&](const char* fmt, auto... xs) -> bool {
    const int bytes =
        std::snprintf(out + write_pos, out_size - write_pos, fmt, xs...);
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

  if (!appendf(
          "mu=%g ax=%g ay=%g rd=%g rp=%g pz=%g px=%g py=%g tag=%hhX",
          static_cast<double>(state.mu), static_cast<double>(state.alpha_x),
          static_cast<double>(state.alpha_y),
          static_cast<double>(state.rd_norm),
          static_cast<double>(state.rp_norm),
          static_cast<double>(state.pz_norm),
          static_cast<double>(state.px_norm),
          static_cast<double>(state.py_norm), static_cast<uint8_t>(state.tag)))
    return write_pos;

  for (size_t i = 0; i < NumCones; ++i) {
    if (!state.predictor[i].active && !state.corrector[i].active) continue;

    const auto& pred = state.predictor[i];
    const auto& corr = state.corrector[i];
    const char* cone_type = pred.active ? pred.cone_type : corr.cone_type;
    if (!appendf("\n  cone[%zu:%s]", i, cone_type)) return write_pos;
    if (!appendf("\n    pred: mu=%g rhs_c=%g pz_rhs=%g%s",
                 static_cast<double>(pred.mu), static_cast<double>(pred.rhs_c),
                 static_cast<double>(pred.pz_rhs),
                 pred.active ? "" : " (inactive)"))
      return write_pos;
    if (pred.active) {
      char buf[1024];
      const auto append_vec = [&](const char* label, const auto& vec,
                                  const size_t size) -> bool {
        const size_t len = array_slice_to_cstr(vec, size, buf, sizeof(buf));
        if (len >= sizeof(buf)) return false;
        return appendf("\n      -> %s: %s", label, buf);
      };
      if (!append_vec("xk", pred.xk, pred.cone_dim)) return write_pos;
      if (!append_vec("yk", pred.yk, pred.cone_dim)) return write_pos;
      if (!append_vec("rhs_c", pred.rhs_c_vec, pred.cone_dim)) return write_pos;
      if (!append_vec("pz", pred.pz, VarDim)) return write_pos;
      if (!append_vec("px", pred.px, pred.cone_dim)) return write_pos;
      if (!append_vec("py", pred.py, pred.cone_dim)) return write_pos;
    }
    if (!appendf("\n    corr: mu=%g rhs_c=%g pz_rhs=%g%s",
                 static_cast<double>(corr.mu), static_cast<double>(corr.rhs_c),
                 static_cast<double>(corr.pz_rhs),
                 corr.active ? "" : " (inactive)"))
      return write_pos;
    if (corr.active) {
      char buf[1024];
      const auto append_vec = [&](const char* label, const auto& vec,
                                  const size_t size) -> bool {
        const size_t len = array_slice_to_cstr(vec, size, buf, sizeof(buf));
        if (len >= sizeof(buf)) return false;
        return appendf("\n      -> %s: %s", label, buf);
      };
      if (!append_vec("xk", corr.xk, corr.cone_dim)) return write_pos;
      if (!append_vec("yk", corr.yk, corr.cone_dim)) return write_pos;
      if (!append_vec("rhs_c", corr.rhs_c_vec, corr.cone_dim)) return write_pos;
      if (!append_vec("pz", corr.pz, VarDim)) return write_pos;
      if (!append_vec("px", corr.px, corr.cone_dim)) return write_pos;
      if (!append_vec("py", corr.py, corr.cone_dim)) return write_pos;
    }
  }

  return write_pos;
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
template <linalg::FieldLike T, size_t NumCones, size_t MaxConeDim, size_t VarDim,
          size_t N>
inline size_t trace_to_cstr(const size_t& num_iter,
                            const std::array<
                                IterState<T, NumCones, MaxConeDim, VarDim>, N>&
                                trace,
                            char* out, const size_t& out_size) {
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
      "tag=%u stalled=%d num_iter=%zu mu=%g convex_obj=%g dual_obj=%g rd=%g "
      "rp=%g\n",
      static_cast<unsigned>(res.tag), res.stalled, res.num_iter,
      static_cast<double>(res.mu), static_cast<double>(res.convex_obj),
      static_cast<double>(res.dual_obj), static_cast<double>(rd_norm),
      static_cast<double>(rp_norm));
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
