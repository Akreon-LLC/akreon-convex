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

  convex/                   DCP atoms, conic canonicalization, and PD-IPM
  control/                  Plant models, controller synthesis, observers
  dynamics/                 Dynamical-system abstractions
  linalg/                   Fixed-size vector, matrix, spectral, and operator APIs
    affine/                 Affine maps, sets, traits, and transforms
    matrix/                 Matrix container and matrix traits
    scalar/                 Scalar and field traits
    space/                  Space descriptors and coordinate maps
    vector/                 Vector container and vector traits
  config/                   SDK assertions and solver configuration
```

## Compile-Time Model

Most SDK APIs encode dimensions in types such as `linalg::Matrix<T, R, C>`,
`linalg::Vector<T, N>`, and spaces derived from those shapes. Control models
compose `StateSpace` with typed `PortLayout` definitions, while convex problems
compose variables, atoms, inequalities, cone blocks, and solver configuration.

The SDK's CMake project builds its verification suite; it does not currently
export a library target for downstream linkage. C++ consumers use
`core/include` as an include root and select an umbrella header:

- `math.hpp` for linear algebra and convex optimization
- `control.hpp` for control models and synthesis
- `dynamics.hpp` for dynamical systems

## Linear Algebra Taxonomy

The linalg types use focused contracts rather than a universal mapping
abstraction:

| Construct | Meaning |
|---|---|
| `Vector` / `Matrix` | Fixed-size mathematical object containers |
| `coordinate_map<Space>` | Converts space elements to and from coordinates |
| `AffineMap` | Affine transformation `x -> A*x + b` |
| `LinearOperator` | Linear action with explicit domain and codomain types |
| `StandardInnerProduct` | Field-aware pairing on a vector domain |
| `NormFunction` | Real-valued magnitude functional |
| Factorizations | Algorithms or representations, not mathematical spaces |

These contracts deliberately remain separate. A matrix or vector can
participate in linear algebra without being assigned a coordinate-space policy.
`coordinate_map` is reserved for representation conversion, while
`operator_traits` describes the types accepted and returned by linear action.

### Inner products

`inner_product(x, y)` is the single vector-pairing operation. It delegates to
`StandardInnerProduct<Vec>` and follows the scalar field:

```text
real:     <x, y> = sum(x(i) * y(i))
complex:  <x, y> = sum(conj(x(i)) * y(i))
```

The complex convention conjugates the first argument, so `<x, x>` is real and
nonnegative. The former explicit pattern `dot(conjugate(x), y)` is therefore
written `inner_product(x, y)`; there is no separate `dot` operation with
different complex semantics.

`StandardInnerProduct` exposes `domain_type`, `scalar_type`, and
`codomain_type`. `inner_product_traits` provides the corresponding normalized
aliases and removes cv/ref qualification before specialization lookup.

### Norm functions

`L1`, `L2`, `Linf`, and `Fro` use the same associated-type vocabulary:

- `norm_domain_t<Norm>` is the vector or matrix being measured.
- `norm_scalar_t<Norm>` is the domain's real or complex field scalar.
- `norm_codomain_t<Norm>` is the associated real scalar.
- `norm_dim_v<Norm>` is the finite coordinate dimension.

`NormFunction` validates both trait registration and the callable result. The
Frobenius norm accepts rectangular field matrices as well as square matrices.

### Linear operators

`operator_traits` describes `scalar_type`, `domain_type`, `codomain_type`, and
the matrix representation when one exists. For `Matrix<T, Rows, Cols>`, the
domain is `Vector<T, Cols>` and the codomain is `Vector<T, Rows>`.

`LinearOperator` verifies application against those associated types.
`EndomorphicOperator` additionally requires identical domain and codomain
types. Factorizations and algorithmic operator wrappers do not receive
`space_traits` merely because they store or act on matrices; genuine spaces
retain the separate element/coordinate contract.

## akreon-native Integration

`akreon-native` is the runtime-facing consumer of the SDK. Its build sets
`AKREON_SDK_INCLUDE_DIR` to the SDK's `core/include` directory and compiles the
selected SDK templates directly into its native C library.

```text
control case manifests
        |
        v
generated static case registry
        |
        v
akreon-sdk control templates
        |
        v
akreon-native C ABI -> nanobind -> Python
```

The manifest-generated registry bridges runtime requests to the finite set of
dimensions and control families compiled into the native binary. The SDK keeps
the mathematical and solver types compile-time explicit; `akreon-native` owns
runtime validation, dispatch, ABI types, Python array conversion, and wheel
packaging. Generated customer wheels contain the compiled native surface, so
their users do not need an SDK checkout or CMake toolchain.

This separation also keeps deployment and language-binding concerns out of the
SDK. Adding a native runtime surface normally consists of defining or exposing
the SDK operation here, then adding its validation, dispatch, and binding in
`akreon-native`.

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
