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
#ifdef AKREON_REPORT_OUTPUT
#include <filesystem>
#include <fstream>
#include <iostream>
#endif

#include "config/assert.hpp"
#include "config/solver_config.hpp"
#include "convex/cone_traits.hpp"
#include "convex/solver/pd_info.hpp"
#include "convex/solver_traits.hpp"
#include "linalg/norms/vector_norms.hpp"

namespace detail {

/// @brief debugging information
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

template <size_t N, typename... Args>
inline bool append_format(char* out, const size_t out_size, size_t& write_pos,
                          const char (&fmt)[N], Args... xs) {
  int bytes{0};
  if constexpr (sizeof...(xs) == 0) {
    bytes = std::snprintf(out + write_pos, out_size - write_pos, "%s", fmt);
  } else {
    bytes = std::snprintf(out + write_pos, out_size - write_pos, fmt, xs...);
  }
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
}

inline constexpr const char* solver_result_to_cstr(
    const config::SolverResult tag) {
  switch (tag) {
    case config::SolverResult::INFEASIBLE_OR_UNBOUNDED:
      return "INFEASIBLE_OR_UNBOUNDED";
    case config::SolverResult::OPTIMAL:
      return "OPTIMAL";
    case config::SolverResult::PRIMAL_FEASIBLE:
      return "PRIMAL_FEASIBLE";
    case config::SolverResult::DUAL_FEASIBLE:
      return "DUAL_FEASIBLE";
    case config::SolverResult::PRIMAL_DUAL_FEASIBLE:
      return "PRIMAL_DUAL_FEASIBLE";
    case config::SolverResult::PRIMAL_OPTIMAL:
      return "PRIMAL_OPTIMAL";
    case config::SolverResult::DUAL_OPTIMAL:
      return "DUAL_OPTIMAL";
    case config::SolverResult::INFEASIBLE_OPTIMAL:
      return "INFEASIBLE_OPTIMAL";
  }
  return "UNKNOWN";
}

inline constexpr const char* numerical_event_to_cstr(
    const linalg::NumericalEvent event) {
  switch (event) {
    case linalg::NumericalEvent::NONE:
      return "NONE";
    case linalg::NumericalEvent::RANK_DEFICIENCY:
      return "RANK_DEFICIENCY";
    case linalg::NumericalEvent::INDEFINITE_FACTORIZATION:
      return "INDEFINITE_FACTORIZATION";
    case linalg::NumericalEvent::NEAR_SINGULAR_TRIANGULAR_SOLVE:
      return "NEAR_SINGULAR_TRIANGULAR_SOLVE";
    case linalg::NumericalEvent::NUMERICAL_SAFEGUARD:
      return "NUMERICAL_SAFEGUARD";
    case linalg::NumericalEvent::DB_MAX_ITERS_REACHED:
      return "DB_MAX_ITERS_REACHED";
    case linalg::NumericalEvent::FACTORIZATION_FAILED:
      return "FACTORIZATION_FAILED";
    case linalg::NumericalEvent::EVD_FAILED:
      return "EVD_FAILED";
  }
  return "UNKNOWN";
}

/**
 * @brief IterState
 */
template <linalg::FieldLike T, size_t MaxConeDim, size_t VarDim, size_t EqDim>
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
  std::array<T, EqDim> pnu{};
  size_t cone_dim{};
  const char* cone_type{"unknown"};
};

/**
 * @brief IterState
 */
template <linalg::FieldLike T, size_t NumCones, size_t MaxConeDim,
          size_t VarDim, size_t EqDim>
struct IterState {
  T mu{};        ///< normalized duality gap
  T alpha_x{};   ///< max step to dK size
  T alpha_y{};   ///< max step to dK^* size
  T rd_norm{};   ///< dual norm residual size
  T rp_norm{};   ///< primal norm residual size
  T req_norm{};  ///< equality norm residual size
  T pz_norm{};   ///< newton step in z \in R^n size
  T pnu_norm{};  ///< newton step in nu size
  T px_norm{};   ///< newton step in x \in K size
  T py_norm{};   ///< newton step in y \in K^* size
  std::array<PredictorCorrectorConeTrace<T, MaxConeDim, VarDim, EqDim>,
             NumCones>
      predictor{};
  std::array<PredictorCorrectorConeTrace<T, MaxConeDim, VarDim, EqDim>,
             NumCones>
      corrector{};
  config::SolverResult tag{};  ///< tag of current iterate

  constexpr void reset_pc_trace() {
    for (size_t i = 0; i < NumCones; ++i) {
      predictor[i] =
          PredictorCorrectorConeTrace<T, MaxConeDim, VarDim, EqDim>{};
      corrector[i] =
          PredictorCorrectorConeTrace<T, MaxConeDim, VarDim, EqDim>{};
    }
  }
};

/**
 * @brief array slice to string
 *
 * @param v (vector)
 * @param size
 * @param out (buffer)
 * @param out_size
 *
 * @return pointer
 */
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
template <linalg::FieldLike T, size_t NumCones, size_t MaxConeDim,
          size_t VarDim, size_t EqDim>
inline size_t iter_state_to_cstr(
    const IterState<T, NumCones, MaxConeDim, VarDim, EqDim>& state, char* out,
    const size_t out_size) {
  AK_ASSERT(out != nullptr && "iter_state_to_cstr requires non-null output");
  if (out == nullptr || out_size == 0) return 0;

  size_t write_pos{0};

  if (!append_format(
          out, out_size, write_pos,
          "mu=%g ax=%g ay=%g rd=%g rp=%g req=%g pz=%g pnu=%g px=%g py=%g "
          "tag=%hhX",
          static_cast<double>(state.mu), static_cast<double>(state.alpha_x),
          static_cast<double>(state.alpha_y),
          static_cast<double>(state.rd_norm),
          static_cast<double>(state.rp_norm),
          static_cast<double>(state.req_norm),
          static_cast<double>(state.pz_norm),
          static_cast<double>(state.pnu_norm),
          static_cast<double>(state.px_norm),
          static_cast<double>(state.py_norm), static_cast<uint8_t>(state.tag)))
    return write_pos;

  for (size_t i = 0; i < NumCones; ++i) {
    const auto& pred = state.predictor[i];
    const auto& corr = state.corrector[i];
    const bool has_pred = pred.cone_dim != 0;
    const bool has_corr = corr.cone_dim != 0;
    if (!has_pred && !has_corr) continue;

    const char* cone_type = has_pred ? pred.cone_type : corr.cone_type;
    if (!append_format(out, out_size, write_pos, "\n  cone[%zu:%s]", i,
                       cone_type))
      return write_pos;
    if (!append_format(
            out, out_size, write_pos, "\n    pred: mu=%g rhs_c=%g pz_rhs=%g%s",
            static_cast<double>(pred.mu), static_cast<double>(pred.rhs_c),
            static_cast<double>(pred.pz_rhs), has_pred ? "" : " (inactive)"))
      return write_pos;
    if (has_pred) {
      char buf[2048];
      const auto append_vec = [&](const char* label, const auto& vec,
                                  const size_t size) -> bool {
        const size_t len = array_slice_to_cstr(vec, size, buf, sizeof(buf));
        if (len >= sizeof(buf)) return false;
        return append_format(out, out_size, write_pos, "\n      -> %s: %s",
                             label, buf);
      };
      if (!append_vec("xk", pred.xk, pred.cone_dim)) return write_pos;
      if (!append_vec("yk", pred.yk, pred.cone_dim)) return write_pos;
      if (!append_vec("rhs_c", pred.rhs_c_vec, pred.cone_dim)) return write_pos;
      if (!append_vec("pz", pred.pz, VarDim)) return write_pos;
      if constexpr (EqDim > 0) {
        if (!append_vec("pnu", pred.pnu, EqDim)) return write_pos;
      }
      if (!append_vec("px", pred.px, pred.cone_dim)) return write_pos;
      if (!append_vec("py", pred.py, pred.cone_dim)) return write_pos;
    }
    if (!append_format(
            out, out_size, write_pos, "\n    corr: mu=%g rhs_c=%g pz_rhs=%g%s",
            static_cast<double>(corr.mu), static_cast<double>(corr.rhs_c),
            static_cast<double>(corr.pz_rhs), has_corr ? "" : " (inactive)"))
      return write_pos;
    if (has_corr) {
      char buf[2048];
      const auto append_vec = [&](const char* label, const auto& vec,
                                  const size_t size) -> bool {
        const size_t len = array_slice_to_cstr(vec, size, buf, sizeof(buf));
        if (len >= sizeof(buf)) return false;
        return append_format(out, out_size, write_pos, "\n      -> %s: %s",
                             label, buf);
      };
      if (!append_vec("xk", corr.xk, corr.cone_dim)) return write_pos;
      if (!append_vec("yk", corr.yk, corr.cone_dim)) return write_pos;
      if (!append_vec("rhs_c", corr.rhs_c_vec, corr.cone_dim)) return write_pos;
      if (!append_vec("pz", corr.pz, VarDim)) return write_pos;
      if constexpr (EqDim > 0) {
        if (!append_vec("pnu", corr.pnu, EqDim)) return write_pos;
      }
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
template <linalg::FieldLike T, size_t NumCones, size_t MaxConeDim,
          size_t VarDim, size_t EqDim, size_t N>
inline size_t trace_to_cstr(
    const size_t& num_iter,
    const std::array<IterState<T, NumCones, MaxConeDim, VarDim, EqDim>, N>&
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
 * @brief vector to json cstring
 *
 * @param vector
 * @param buffer char string
 * @param buffer size
 *
 * @return write position
 */
template <linalg::VectorLike Vec>
inline size_t vec_to_json_cstr(const Vec& v, char* out,
                               const size_t& out_size) {
  return vec_to_cstr(v, out, out_size);
}

/**
 * @brief matrix to json cstring
 *
 * @param matrix
 * @param buffer char string
 * @param buffer size
 *
 * @return write position
 */
template <linalg::MatrixLike Mat>
inline size_t mat_to_json_cstr(const Mat& M, char* out,
                               const size_t& out_size) {
  AK_ASSERT(out != nullptr && "mat_to_json_cstr requires non-null output");
  if (out == nullptr || out_size == 0) return 0;

  constexpr size_t Rows = linalg::mat_rows_v<Mat>;
  constexpr size_t Cols = linalg::mat_cols_v<Mat>;

  size_t write_pos{0};
  if (!append_format(out, out_size, write_pos, "[")) return write_pos;
  for (size_t i = 0; i < Rows; ++i) {
    if (!append_format(out, out_size, write_pos, "[")) return write_pos;
    for (size_t j = 0; j < Cols; ++j) {
      if (j + 1 == Cols) {
        if (!append_format(out, out_size, write_pos, "%g",
                           static_cast<double>(M(i)(j))))
          return write_pos;
      } else {
        if (!append_format(out, out_size, write_pos, "%g, ",
                           static_cast<double>(M(i)(j))))
          return write_pos;
      }
    }
    if (i + 1 == Rows) {
      if (!append_format(out, out_size, write_pos, "]")) return write_pos;
    } else {
      if (!append_format(out, out_size, write_pos, "],")) return write_pos;
    }
  }
  if (!append_format(out, out_size, write_pos, "]")) return write_pos;
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
template <linalg::VectorLike Var, linalg::VectorLike ConeVar,
          linalg::VectorLike NuVar>
inline size_t result_to_cstr(const PrimalDualResult<Var, ConeVar, NuVar>& res,
                             char* out, const size_t& out_size) {
  AK_ASSERT(out != nullptr && "result_to_cstr requires non-null output");
  if (out == nullptr || out_size == 0) return 0;

  size_t write_pos{0};
  const auto append_text = [&](const char* text) -> bool {
    return append_format(out, out_size, write_pos, "%s", text);
  };

  const auto rd_norm = linalg::L2<decltype(res.rd)>()(res.rd);
  const auto rp_norm = linalg::L2<decltype(res.rp)>()(res.rp);
  const auto req_norm = linalg::L2<decltype(res.req)>()(res.req);

  const int header = std::snprintf(
      out + write_pos, out_size - write_pos,
      "tag=%u stalled=%d event=%hhu num_iter=%zu mu=%g convex_obj=%g "
      "dual_obj=%g "
      "rd=%g "
      "rp=%g "
      "req=%g\n",
      static_cast<unsigned>(res.tag), res.stalled,
      static_cast<uint8_t>(res.event), res.num_iter,
      static_cast<double>(res.mu), static_cast<double>(res.convex_obj),
      static_cast<double>(res.dual_obj), static_cast<double>(rd_norm),
      static_cast<double>(rp_norm), static_cast<double>(req_norm));
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
  write_pos +=
      vec_to_cstr(res.iterate.z, out + write_pos, out_size - write_pos);
  if (write_pos >= out_size - 1) return out_size - 1;
  out[write_pos++] = '\n';
  out[write_pos] = '\0';

  if (!append_text("x=")) return write_pos;
  write_pos +=
      vec_to_cstr(res.iterate.x, out + write_pos, out_size - write_pos);
  if (write_pos >= out_size - 1) return out_size - 1;
  out[write_pos++] = '\n';
  out[write_pos] = '\0';

  if (!append_text("y=")) return write_pos;
  write_pos +=
      vec_to_cstr(res.iterate.y, out + write_pos, out_size - write_pos);
  if (write_pos >= out_size - 1) return out_size - 1;
  out[write_pos++] = '\n';
  out[write_pos] = '\0';

  if (!append_text("nu=")) return write_pos;
  write_pos +=
      vec_to_cstr(res.iterate.nu, out + write_pos, out_size - write_pos);
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

  if (!append_text("req=")) return write_pos;
  write_pos += vec_to_cstr(res.req, out + write_pos, out_size - write_pos);
  if (write_pos >= out_size - 1) return out_size - 1;
  out[write_pos++] = '\n';
  out[write_pos] = '\0';

  return write_pos;
}

/**
 * @brief primal-dual result to json cstring
 *
 * @param result
 * @param buffer char string
 * @param buffer size
 *
 * @return write position
 */
template <linalg::VectorLike Var, linalg::VectorLike ConeVar,
          linalg::VectorLike NuVar>
inline size_t result_to_json_cstr(
    const PrimalDualResult<Var, ConeVar, NuVar>& res, char* out,
    const size_t& out_size) {
  AK_ASSERT(out != nullptr && "result_to_json_cstr requires non-null output");
  if (out == nullptr || out_size == 0) return 0;

  size_t write_pos{0};
  const auto append_json_vec = [&](const char* key, const auto& vec) -> bool {
    char buf[8192];
    const auto len = vec_to_json_cstr(vec, buf, sizeof(buf));
    if (len >= sizeof(buf)) return false;
    return append_format(out, out_size, write_pos, "\"%s\":%s", key, buf);
  };

  const auto rd_norm = linalg::L2<decltype(res.rd)>()(res.rd);
  const auto rp_norm = linalg::L2<decltype(res.rp)>()(res.rp);
  const auto req_norm = linalg::L2<decltype(res.req)>()(res.req);

  if (!append_format(out, out_size, write_pos, "{")) return write_pos;
  if (!append_format(out, out_size, write_pos, "\"tag\":\"%s\",",
                     solver_result_to_cstr(res.tag)))
    return write_pos;
  if (!append_format(out, out_size, write_pos, "\"tag_code\":%u,",
                     static_cast<unsigned>(res.tag)))
    return write_pos;
  if (!append_format(out, out_size, write_pos, "\"event\":\"%s\",",
                     numerical_event_to_cstr(res.event)))
    return write_pos;
  if (!append_format(out, out_size, write_pos, "\"event_code\":%u,",
                     static_cast<unsigned>(res.event)))
    return write_pos;
  if (!append_format(out, out_size, write_pos, "\"stalled\":%s,",
                     res.stalled ? "true" : "false"))
    return write_pos;
  if (!append_format(out, out_size, write_pos, "\"num_iter\":%zu,",
                     res.num_iter))
    return write_pos;
  if (!append_format(out, out_size, write_pos, "\"mu\":%g,",
                     static_cast<double>(res.mu)))
    return write_pos;
  if (!append_format(out, out_size, write_pos, "\"convex_obj\":%g,",
                     static_cast<double>(res.convex_obj)))
    return write_pos;
  if (!append_format(out, out_size, write_pos, "\"dual_obj\":%g,",
                     static_cast<double>(res.dual_obj)))
    return write_pos;

  if (!append_format(out, out_size, write_pos, "\"norms\":{")) return write_pos;
  if (!append_format(out, out_size, write_pos, "\"rd\":%g,",
                     static_cast<double>(rd_norm)))
    return write_pos;
  if (!append_format(out, out_size, write_pos, "\"rp\":%g,",
                     static_cast<double>(rp_norm)))
    return write_pos;
  if (!append_format(out, out_size, write_pos, "\"req\":%g}",
                     static_cast<double>(req_norm)))
    return write_pos;

  if (!append_format(out, out_size, write_pos, ",\"iterate\":{"))
    return write_pos;
  if (!append_json_vec("z", res.iterate.z)) return write_pos;
  if (!append_format(out, out_size, write_pos, ",")) return write_pos;
  if (!append_json_vec("x", res.iterate.x)) return write_pos;
  if (!append_format(out, out_size, write_pos, ",")) return write_pos;
  if (!append_json_vec("y", res.iterate.y)) return write_pos;
  if (!append_format(out, out_size, write_pos, ",")) return write_pos;
  if (!append_json_vec("nu", res.iterate.nu)) return write_pos;
  if (!append_format(out, out_size, write_pos, "}")) return write_pos;

  if (!append_format(out, out_size, write_pos, ",\"residuals\":{"))
    return write_pos;
  if (!append_json_vec("rd", res.rd)) return write_pos;
  if (!append_format(out, out_size, write_pos, ",")) return write_pos;
  if (!append_json_vec("rp", res.rp)) return write_pos;
  if (!append_format(out, out_size, write_pos, ",")) return write_pos;
  if (!append_json_vec("req", res.req)) return write_pos;
  if (!append_format(out, out_size, write_pos, "}")) return write_pos;

  if (!append_format(out, out_size, write_pos, "}")) return write_pos;
  return write_pos;
}

/**
 * @brief primal-dual result to json cstring with additional matrix field
 *
 * @param result
 * @param field_name json key for the matrix field
 * @param matrix additional matrix payload such as synthesized K or L
 * @param buffer char string
 * @param buffer size
 *
 * @return write position
 */
template <linalg::VectorLike Var, linalg::VectorLike ConeVar,
          linalg::VectorLike NuVar, linalg::MatrixLike Mat>
inline size_t result_to_json_cstr(
    const PrimalDualResult<Var, ConeVar, NuVar>& res, const char* field_name,
    const Mat& matrix, char* out, const size_t& out_size) {
  AK_ASSERT(field_name != nullptr &&
            "result_to_json_cstr requires non-null field name");
  AK_ASSERT(out != nullptr && "result_to_json_cstr requires non-null output");
  if (field_name == nullptr || out == nullptr || out_size == 0) return 0;

  char base_buf[64000];
  const auto base_len = result_to_json_cstr(res, base_buf, sizeof(base_buf));
  if (base_len == 0 || base_len >= sizeof(base_buf)) return 0;

  char mat_buf[8192];
  const auto mat_len = mat_to_json_cstr(matrix, mat_buf, sizeof(mat_buf));
  if (mat_len == 0 || mat_len >= sizeof(mat_buf)) return 0;

  size_t write_pos{0};
  if (base_len < 1 || base_buf[base_len - 1] != '}') return 0;
  if (!append_format(out, out_size, write_pos, "%.*s",
                     static_cast<int>(base_len - 1), base_buf))
    return write_pos;
  if (!append_format(out, out_size, write_pos, ",\"%s\":%s}", field_name,
                     mat_buf))
    return write_pos;
  return write_pos;
}

#ifdef AKREON_REPORT_OUTPUT
template <linalg::VectorLike Var, linalg::VectorLike ConeVar,
          linalg::VectorLike NuVar>
inline void print_result(std::ostream& os,
                         const PrimalDualResult<Var, ConeVar, NuVar>& res) {
  char buf[64000];
  result_to_cstr(res, buf, sizeof(buf));
  os << buf;
}

template <linalg::FieldLike T, size_t NumCones, size_t MaxConeDim,
          size_t VarDim, size_t EqDim, size_t N>
inline void print_trace(
    std::ostream& os, const size_t& num_iter,
    const std::array<IterState<T, NumCones, MaxConeDim, VarDim, EqDim>, N>&
        trace) {
  char buf[64000];
  trace_to_cstr(num_iter, trace, buf, sizeof(buf));
  os << buf;
}

template <linalg::VectorLike Var, linalg::VectorLike ConeVar,
          linalg::VectorLike NuVar>
inline void report_result_json(
    std::ostream& os, const PrimalDualResult<Var, ConeVar, NuVar>& res) {
  char buf[64000];
  result_to_json_cstr(res, buf, sizeof(buf));
  os << buf;
}

template <linalg::VectorLike Var, linalg::VectorLike ConeVar,
          linalg::VectorLike NuVar, linalg::MatrixLike Mat>
inline void report_result_json(
    std::ostream& os, const PrimalDualResult<Var, ConeVar, NuVar>& res,
    const char* field_name, const Mat& matrix) {
  char buf[64000];
  result_to_json_cstr(res, field_name, matrix, buf, sizeof(buf));
  os << buf;
}

template <linalg::VectorLike Var, linalg::VectorLike ConeVar,
          linalg::VectorLike NuVar>
inline bool write_result_json(
    const std::filesystem::path& path,
    const PrimalDualResult<Var, ConeVar, NuVar>& res) {
  std::ofstream out(path);
  if (!out.is_open()) return false;
  report_result_json(out, res);
  return static_cast<bool>(out);
}

template <linalg::VectorLike Var, linalg::VectorLike ConeVar,
          linalg::VectorLike NuVar, linalg::MatrixLike Mat>
inline bool write_result_json(
    const std::filesystem::path& path,
    const PrimalDualResult<Var, ConeVar, NuVar>& res, const char* field_name,
    const Mat& matrix) {
  std::ofstream out(path);
  if (!out.is_open()) return false;
  report_result_json(out, res, field_name, matrix);
  return static_cast<bool>(out);
}
#endif

}  // namespace convex
