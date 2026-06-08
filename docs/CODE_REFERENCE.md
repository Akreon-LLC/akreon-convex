# akreon-sdk Code Reference

This document orients the public SDK headers. The SDK is intentionally
template-heavy: dimensions, algebraic spaces, cone blocks, and control problem
definitions are compile-time types so generated users can keep problem
structure explicit and avoid hidden runtime allocation in real-time paths.

## Source Layout

```text
core/include/
  math.hpp                  Top-level math umbrella
  control.hpp               Top-level control umbrella
  dynamics.hpp              Top-level dynamics umbrella

  affine/                   Affine maps, sets, traits, and transforms
  convex/                   DCP atoms, conic canonicalization, and PD-IPM
  control/                  Plant models, controller synthesis, observers
  dynamics/                 Dynamical-system abstractions
  linalg/                   Fixed-size vector, matrix, spectral, and operator APIs
  config/                   SDK assertions and solver configuration
```

## Compile-Time Model

Most SDK APIs encode dimensions in types such as `linalg::Matrix<T, R, C>`,
`linalg::Vector<T, N>`, and spaces derived from those shapes. Control models
compose `StateSpace` with typed `PortLayout` definitions, while convex problems
compose variables, atoms, inequalities, cone blocks, and solver configuration.

## Convex Canonicalization Flow

1. User-facing atoms validate DCP composition and domain/codomain compatibility.
2. Atoms canonicalize into affine cone-coordinate maps plus emitted cone tuples.
3. `CanonicalFormBuilder` accumulates cone rows, equality rows, and auxiliary
   variables under a stable column layout.
4. `ConvexProblem` and solver components consume the canonical form to run the
   primal-dual interior-point method.

## Control Synthesis Flow

1. `StateSpace` captures continuous or discrete plant dynamics.
2. `PlantModel` selects role-tagged input and output ports.
3. Feedback and observer synthesis structs build convex constraints for the
   selected synthesis family.
4. Gain-recovery helpers map solver variables back to controller gains.

## Documentation Generation

A Doxygen config is available at `Doxyfile`.

```bash
doxygen Doxyfile
```

The generated HTML output is written under `docs/doxygen/html`.
