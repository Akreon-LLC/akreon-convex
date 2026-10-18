# Akreon Convex Architecture

This guide describes expression assembly, canonicalization, and solver data flow.
For supported problem classes and future work, see the [support roadmap](ROADMAP.md).

## Source organization

Paths are relative to `include/akreon/convex/`.

| Path | Responsibility |
| --- | --- |
| `atom.hpp`, `variable.hpp`, `convex_traits.hpp` | Typed expressions, variables, and DCP contracts |
| `functions/`, `composed/` | Atoms and compositions, including quadratic forms |
| `sets/`, `inequality.hpp`, `intersection.hpp`, `sum.hpp` | Constraints and expression combinations |
| `canonical_emission.hpp`, `canonical_form_builder.hpp` | Local recipes and shared cone/equality layouts |
| `cones/`, `cone_traits.hpp`, `cone_block.hpp` | Cone operations and product blocks |
| `convex_problem.hpp` | Problem assembly and objectives |
| `solver/`, `solver_traits.hpp`, `solver_config.hpp` | Primal-dual iteration, results, and configuration |
| `canonical_trace.hpp`, `debug/` | Canonicalization traces and diagnostic formatting |

## Data flow

1. Variables and atoms encode dimensions, domains, curvature, and composition
   in template types. Constraints and objectives form a typed problem.
2. Atoms emit affine cone maps and auxiliary/equality requirements. The builder
   merges them into a shared layout of decision and auxiliary coordinates.
3. The primal-dual solver consumes that canonical form, evaluates residuals,
   forms Newton/Schur systems, and applies predictor/corrector updates.
4. Results carry termination status, numerical events, and optional traces.
   Callers must inspect those results; a valid model need not have an optimum.

## Ownership and reference

`convex` owns modeling and solving; `conic` owns cone/building infrastructure;
`config` holds solver settings. Convex owns optimization domains and cone
membership; Linalg owns generic matrices, spaces, affine maps,
and numerical kernels. Convex has no dependency on Control or Dynamics.

Doxygen's namespace, class, and file indexes provide API contracts and algorithm
flow from the headers.
