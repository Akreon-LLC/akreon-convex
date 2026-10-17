# Solver Numerical Hardening Notes

This note tracks numerical-hardening work exposed by the dynamic-weighted
quarter-car H∞ tutorial. The issue is not currently treated as a correctness
failure in the core solver. It is a conditioning and coefficient-canonicalization
concern that can make recovered controller gains sensitive to machine-precision
construction noise.

## Observed Case

The quarter-car dynamic-weighted H∞ state-feedback example augments the nominal
4-state plant with first-order performance-weight filter states. After
nondimensionalization, the scaled realization is order-one, but the resulting
SDP is sensitive to perturbations around double-precision construction noise.

Equivalent algebraic construction paths produced scaled matrices differing only
near machine precision, yet the recovered H∞ gain could follow different
numerical paths:

- one path recovered a fragile gain whose post-synthesis
  `||w_bar -> z_dynamic||_inf` estimate exceeded solver `gamma`,
- another path recovered a well-conditioned gain that satisfied the same
  post-check.

The local tutorial workaround is to canonicalize the already-scaled matrices at
double-precision resolution before synthesis. This acts like coefficient
cleanup/presolve: it removes fake weak couplings and restores the intended
sparsity pattern. It does not replace scaling, regularization, or post-solve
verification.

## Current Solver Regularization Path

The primal-dual IPM currently regularizes immediately before factorization of
the accumulated block:

```cpp
const auto [M_blk, block_event] =
    accumulate_block(BlockMat{}, F, K, x, y);

const auto M_ldl = linalg::LDL(regularize(M_blk, A, config.reg));
const auto [eq_opt, coupling_event] = make_equality_coupling(M_ldl, A);
```

The current knobs are:

- `RegularizationConfig::global_eps`: scaled diagonal bump applied globally,
- `RegularizationConfig::eq_eps`: larger bump for weak equality-active
  directions.

The native bridge maps these from:

- `SolverConfig.global_reg_eps`,
- `SolverConfig.eq_reg_eps`.

In the quarter-car case, sweeping these values helped only in narrow,
non-monotonic regions. Equality-direction regularization did not materially
change that example. That suggests the first-order issue is coefficient
canonicalization / problem conditioning, while KKT regularization remains a
secondary hardening lever.

## Hardening Task List

### 1. Coefficient cleanup / canonicalization pass

Add an optional canonicalization step for canonical conic data or user-supplied
problem matrices:

- drop entries below an absolute or relative threshold,
- preserve intended sparsity patterns,
- report how many coefficients were dropped,
- keep the pass opt-in or diagnostic by default until tested broadly.

This is analogous to solver presolve coefficient cleanup. It should operate
after user scaling/canonicalization has produced the numeric problem data, not
as a substitute for physical nondimensionalization.

### 2. Solver diagnostics for conditioning

Expose additional diagnostics in `PrimalDualResult` or native result surfaces:

- LDL/factorization health,
- regularization magnitudes used,
- near-singular triangular-solve events,
- residuals at return,
- approximate condition indicators where feasible.

The goal is to tell users whether a returned solution is merely feasible under
tolerances or also numerically robust for recovery.

### 3. H∞ gain recovery

See akreon-control `docs/GAIN_RECOVERY.md`.

### 4. Adaptive regularization policy

Replace fixed regularization-only behavior with an adaptive path:

- start from configured `global_eps` / `eq_eps`,
- increase regularization when LDL or coupling construction is weak,
- record the applied value in solver diagnostics,
- avoid silent over-regularization that materially changes the problem.

Regularization should remain explicit and observable.

### 5. Feasibility re-solve or post-certification path

For robust-control workflows, add a way to validate or re-solve for a target
bound:

- solve minimization to get candidate `gamma`,
- recover controller,
- estimate or certify achieved closed-loop norm,
- optionally run a feasibility solve at `gamma_target > achieved_norm` to
  recover a better-conditioned certificate.

This would make tutorial and production workflows less dependent on the first
minimal-gamma recovery path.

### 6. Regression example

Preserve the quarter-car dynamic-weighted H∞ case as a regression once the
generated-native fixture surface is stable enough:

- assert that scaled matrices remain order-one,
- assert synthesis returns a bounded post-check,
- assert no fragile recovery event is reported,
- compare rounded/canonicalized and raw construction paths.

## Working Classification

This is currently classified as:

1. primary: numerical-analysis / problem-conditioning concern,
2. secondary: solver-hardening opportunity,
3. not: a control-theory modeling requirement.

The tutorial-level matrix rounding is acceptable as numerical hygiene for an
educational example, but core hardening should prefer explicit presolve,
diagnostics, adaptive regularization, and recovery validation.
