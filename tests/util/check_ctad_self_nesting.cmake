# Copyright © 2025-2026 Akreon LLC
# SPDX-License-Identifier: Apache-2.0

set(_akreon_scan_roots
  "${SOURCE_DIR}/include/akreon/convex"
  "${SOURCE_DIR}/tests"
)

set(_akreon_patterns
  "Abs|abs_"
  "Exp|exp_"
  "Log|log_"
  "Square|square_"
  "Sqrt|sqrt_"
  "Softplus|softplus_"
  "Max|max_"
  "GeoMean|geo_mean_"
  "HarmonicMean|harmonic_mean_"
  "Huber|huber_"
  "InvPos|inv_pos_"
  "KLDivergence|kl_div_"
  "LogSumExp|log_sum_exp_"
)

set(_akreon_matches)

foreach(_akreon_root IN LISTS _akreon_scan_roots)
  file(GLOB_RECURSE _akreon_files
    LIST_DIRECTORIES false
    "${_akreon_root}/*.hpp"
    "${_akreon_root}/*.cpp"
  )

  foreach(_akreon_file IN LISTS _akreon_files)
    file(STRINGS "${_akreon_file}" _akreon_lines)
    set(_akreon_line_no 0)

    foreach(_akreon_line IN LISTS _akreon_lines)
      math(EXPR _akreon_line_no "${_akreon_line_no} + 1")
      string(REGEX REPLACE "//.*$" "" _akreon_code_line "${_akreon_line}")

      foreach(_akreon_pattern IN LISTS _akreon_patterns)
        string(REPLACE "|" ";" _akreon_parts "${_akreon_pattern}")
        list(GET _akreon_parts 0 _akreon_ctor)
        list(GET _akreon_parts 1 _akreon_var_prefix)

        string(REGEX MATCH
          "(^|[^A-Za-z0-9_])${_akreon_ctor}[ \t]*\\([ \t]*${_akreon_var_prefix}[A-Za-z0-9_]*"
          _akreon_match
          "${_akreon_code_line}"
        )

        if(_akreon_match)
          list(APPEND _akreon_matches
            "${_akreon_file}:${_akreon_line_no}: suspicious self-nesting CTAD '${_akreon_ctor}(...)' on '${_akreon_var_prefix}*'; use explicit template construction")
          break()
        endif()
      endforeach()
    endforeach()
  endforeach()
endforeach()

if(_akreon_matches)
  string(JOIN "\n" _akreon_message ${_akreon_matches})
  message(FATAL_ERROR
    "CTAD self-nesting lint failed.\n"
    "These patterns can trigger copy-deduction instead of true composition.\n"
    "Use explicit template construction for same-family nesting.\n"
    "${_akreon_message}")
endif()
