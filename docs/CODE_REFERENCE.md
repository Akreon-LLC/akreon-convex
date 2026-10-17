# akreon-convex Code Reference

Include `<akreon/convex/convex.hpp>` and link `Akreon::Convex`.

## Convex Canonicalization Flow

1. User-facing atoms validate DCP composition and domain/codomain compatibility.
2. Atoms canonicalize into affine cone-coordinate maps plus emitted cone tuples.
3. `CanonicalFormBuilder` accumulates cone rows, equality rows, and auxiliary
   variables under a stable column layout.
4. `ConvexProblem` and solver components consume the canonical form to run the
   primal-dual interior-point method.

See also
[`SOLVER_NUMERICAL_HARDENING.md`](SOLVER_NUMERICAL_HARDENING.md) for solver
conditioning, coefficient-cleanup, and H∞ gain-recovery hardening notes.
