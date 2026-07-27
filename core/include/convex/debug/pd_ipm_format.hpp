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

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#ifdef AKREON_ENABLE_IOSTREAM
#include <ostream>
#endif

#include "convex/canonical_trace.hpp"
#include "convex/debug/fixed_buffer_writer.hpp"
#include "convex/solver/pd_ipm_result.hpp"
#include "convex/solver/pd_ipm_trace.hpp"
#include "linalg/norms/vector_norms.hpp"

namespace convex::debug {

inline constexpr const char* solver_result_to_cstr(config::SolverResult tag) {
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
    linalg::NumericalEvent event) {
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

inline constexpr const char* canonical_cone_kind_to_cstr(
    CanonicalConeKind kind) {
  switch (kind) {
    case CanonicalConeKind::NONNEG_ORTHANT:
      return "RPLUS";
    case CanonicalConeKind::SECOND_ORDER:
      return "SOC";
    case CanonicalConeKind::POSITIVE_SEMIDEFINITE:
      return "PSD";
    case CanonicalConeKind::EXP:
      return "EXP";
    case CanonicalConeKind::POWER:
      return "POW";
    case CanonicalConeKind::UNKNOWN:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

template <typename Trace>
inline bool append_canonical_trace(FixedBufferWriter& w, const Trace& trace) {
  if (!w.format("canonical_trace: nodes=%zu cones=%zu equalities=%zu\n",
                canonical_trace_num_nodes_v<Trace>,
                canonical_trace_num_cones_v<Trace>,
                canonical_trace_num_equalities_v<Trace>))
    return false;

  std::size_t i = 0;
  bool ok = true;
  std::apply(
      [&](const auto&... nodes) {
        ((ok = ok &&
               w.format("  node[%zu]: id=%zu parent=%zu atom=%u child=%zu\n",
                        i++, nodes.node_id, nodes.parent_id,
                        static_cast<unsigned>(nodes.atom_type),
                        nodes.child_index)),
         ...);
      },
      trace.nodes);
  if (!ok) return false;

  i = 0;
  std::apply(
      [&](const auto&... cones) {
        ((ok = ok &&
               w.format("  cone[%zu]: node=%zu local=%zu global=%zu kind=%s "
                        "offset=%zu dim=%zu\n",
                        i++, cones.node_id, cones.local_cone_index,
                        cones.global_cone_index,
                        canonical_cone_kind_to_cstr(cones.cone_kind),
                        cones.cone_offset, cones.cone_dim)),
         ...);
      },
      trace.cones);
  if (!ok) return false;

  i = 0;
  std::apply(
      [&](const auto&... eqs) {
        ((ok = ok &&
               w.format("  eq[%zu]: node=%zu local=%zu offset=%zu rows=%zu\n",
                        i++, eqs.node_id, eqs.local_eq_index, eqs.row_offset,
                        eqs.row_count)),
         ...);
      },
      trace.equalities);
  return ok;
}

template <linalg::Field T, std::size_t NumCones, typename ConeTuple,
          std::size_t... Is>
inline bool append_cone_steps(FixedBufferWriter& w,
                              const PdIpmStepTrace<T, NumCones>& step,
                              const ConeTuple& cones,
                              std::index_sequence<Is...>, const char* label) {
  auto append_cone_step = [&](const ConeStepTrace<T>& cone_step) {
    return w.format("active=%d mu=%g rhs_c=%g pz_rhs=%g", cone_step.active,
                    static_cast<double>(cone_step.mu),
                    static_cast<double>(cone_step.rhs_c_norm),
                    static_cast<double>(cone_step.pz_rhs_norm));
  };
  bool ok = true;
  ((ok =
        ok &&
        w.format("    %s cone[%zu:%s node=%zu offset=%zu dim=%zu]: ", label, Is,
                 canonical_cone_kind_to_cstr(std::get<Is>(cones).cone_kind),
                 std::get<Is>(cones).node_id, std::get<Is>(cones).cone_offset,
                 std::get<Is>(cones).cone_dim) &&
        append_cone_step(step.cones[Is]) && w.append("\n")),
   ...);
  return ok;
}

template <linalg::Field T, std::size_t NumCones, typename CanonicalTraceT>
inline bool append_iter_trace(FixedBufferWriter& w,
                              const PdIpmIterTrace<T, NumCones>& iter,
                              const CanonicalTraceT& canonical_trace,
                              std::size_t k) {
  if (!w.format(
          "[%zu] mu=%g ax=%g ay=%g rd=%g rp=%g req=%g pz=%g pnu=%g "
          "px=%g py=%g tag=%s event=%s stalled=%d\n",
          k, static_cast<double>(iter.mu), static_cast<double>(iter.alpha_x),
          static_cast<double>(iter.alpha_y), static_cast<double>(iter.rd_norm),
          static_cast<double>(iter.rp_norm), static_cast<double>(iter.req_norm),
          static_cast<double>(iter.pz_norm), static_cast<double>(iter.pnu_norm),
          static_cast<double>(iter.px_norm), static_cast<double>(iter.py_norm),
          solver_result_to_cstr(iter.tag), numerical_event_to_cstr(iter.event),
          iter.stalled))
    return false;

  constexpr std::size_t TraceCones =
      canonical_trace_num_cones_v<CanonicalTraceT>;
  if constexpr (TraceCones == NumCones) {
    if (!append_cone_steps(w, iter.predictor, canonical_trace.cones,
                           std::make_index_sequence<NumCones>{}, "pred"))
      return false;
    if (!append_cone_steps(w, iter.corrector, canonical_trace.cones,
                           std::make_index_sequence<NumCones>{}, "corr"))
      return false;
  } else {
    for (std::size_t i = 0; i < NumCones; ++i) {
      if (!w.format("    pred cone[%zu]: ", i)) return false;
      if (!append_cone_step(w, iter.predictor.cones[i])) return false;
      if (!w.append("\n")) return false;
      if (!w.format("    corr cone[%zu]: ", i)) return false;
      if (!append_cone_step(w, iter.corrector.cones[i])) return false;
      if (!w.append("\n")) return false;
    }
  }

  return true;
}

template <linalg::Field T, std::size_t NumCones, typename CanonicalTraceT>
inline bool append_solver_trace(FixedBufferWriter& w,
                                const PdIpmTrace<T, NumCones>& trace,
                                const CanonicalTraceT& canonical_trace) {
  if (!w.format("solver_trace: iters=%zu max_iters=%zu cones=%zu\n",
                trace.num_iters, PD_IPM_STEPS, NumCones))
    return false;

  const std::size_t used =
      (trace.num_iters < PD_IPM_STEPS) ? trace.num_iters : PD_IPM_STEPS;
  for (std::size_t k = 0; k < used; ++k) {
    if (!append_iter_trace(w, trace.iters[k], canonical_trace, k)) return false;
  }
  return true;
}

template <typename Trace, typename CanonicalTraceT>
inline bool append_solver_trace(FixedBufferWriter& w, const Trace&,
                                const CanonicalTraceT&) {
  return w.append("solver_trace: <empty>\n");
}

template <linalg::VectorContainer Var, linalg::VectorContainer ConeVar,
          linalg::VectorContainer NuVar, typename CanonicalTraceT,
          typename SolverTraceT>
inline bool append_result(
    FixedBufferWriter& w,
    const convex::PrimalDualResult<Var, ConeVar, NuVar, CanonicalTraceT,
                                   SolverTraceT>& res) {
  const auto rd_norm = linalg::L2<decltype(res.rd)>()(res.rd);
  const auto rp_norm = linalg::L2<decltype(res.rp)>()(res.rp);
  const auto req_norm = linalg::L2<decltype(res.req)>()(res.req);

  if (!w.format("tag=%s stalled=%d event=%s num_iter=%zu mu=%g "
                "convex_obj=%g dual_obj=%g rd=%g rp=%g req=%g\n",
                solver_result_to_cstr(res.tag), res.stalled,
                numerical_event_to_cstr(res.event), res.num_iter,
                static_cast<double>(res.mu),
                static_cast<double>(res.convex_obj),
                static_cast<double>(res.dual_obj), static_cast<double>(rd_norm),
                static_cast<double>(rp_norm), static_cast<double>(req_norm)))
    return false;

  if (!w.append("z=")) return false;
  if (!append_vector(w, res.iterate.z)) return false;
  if (!w.append("\n")) return false;
  if (!w.append("x=")) return false;
  if (!append_vector(w, res.iterate.x)) return false;
  if (!w.append("\n")) return false;
  if (!w.append("y=")) return false;
  if (!append_vector(w, res.iterate.y)) return false;
  if (!w.append("\n")) return false;
  if (!w.append("nu=")) return false;
  if (!append_vector(w, res.iterate.nu)) return false;
  if (!w.append("\n")) return false;
  if (!w.append("rd=")) return false;
  if (!append_vector(w, res.rd)) return false;
  if (!w.append("\n")) return false;
  if (!w.append("rp=")) return false;
  if (!append_vector(w, res.rp)) return false;
  if (!w.append("\n")) return false;
  if (!w.append("req=")) return false;
  if (!append_vector(w, res.req)) return false;
  if (!w.append("\n")) return false;

  if (!append_canonical_trace(w, res.canonical_trace)) return false;
  return append_solver_trace(w, res.solver_trace, res.canonical_trace);
}

template <linalg::VectorContainer Var, linalg::VectorContainer ConeVar,
          linalg::VectorContainer NuVar, typename CanonicalTraceT,
          typename SolverTraceT>
inline std::size_t result_to_cstr(
    const convex::PrimalDualResult<Var, ConeVar, NuVar, CanonicalTraceT,
                                   SolverTraceT>& res,
    char* out, std::size_t out_size) {
  FixedBufferWriter w{out, out_size, 0, false};
  w.init();
  append_result(w, res);
  return w.size();
}

#ifdef AKREON_ENABLE_IOSTREAM
template <linalg::VectorContainer Var, linalg::VectorContainer ConeVar,
          linalg::VectorContainer NuVar, typename CanonicalTraceT,
          typename SolverTraceT>
inline std::ostream& operator<<(
    std::ostream& os,
    const convex::PrimalDualResult<Var, ConeVar, NuVar, CanonicalTraceT,
                                   SolverTraceT>& res) {
  char buf[64000];
  result_to_cstr(res, buf, sizeof(buf));
  return os << buf;
}
#endif

} // namespace convex::debug
