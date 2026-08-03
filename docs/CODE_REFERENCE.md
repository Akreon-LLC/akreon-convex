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

### Matrix container

`Matrix<T, Rows, Cols>` is an owning, fixed-size container whose shape is part
of its type. It stores `Rows * Cols` field elements inline in contiguous
row-major order and performs no dynamic allocation. Default construction
value-initializes the elements, and the container API is `constexpr` so matrices
can participate in compile-time contracts and calculations.

Use zero-based two-dimensional indexing:

```cpp
constexpr linalg::Matrix<double, 2, 3> A{
    linalg::Vector<double, 3>{1.0, 2.0, 3.0},
    linalg::Vector<double, 3>{4.0, 5.0, 6.0}};

static_assert(A(0, 2) == 3.0);
static_assert(A(1, 0) == 4.0);
```

The row-vector constructor supports CTAD, so the explicit matrix type may be
omitted when all rows have the same field type and dimension:

```cpp
constexpr auto A = linalg::Matrix{
    linalg::Vector<double, 2>{1.0, 2.0},
    linalg::Vector<double, 2>{3.0, 4.0}};
// A is Matrix<double, 2, 2>.
```

`row(i)` returns a vector copy. `view()` returns a fixed-extent `std::span` over
the contiguous storage, while `begin()` and `end()` traverse the same flattened
row-major order. Element and row access use `AK_ASSERT` for bounds checking.

Conversions between vectors and one-dimensional matrices are explicit through
`as_row_matrix`, `as_col_matrix`, and `as_vector`. A general rectangular matrix
does not implicitly flatten to a vector.

The container itself does not assert symmetry, positive semidefiniteness, or a
coordinate convention. Matrix arithmetic is supplied by the linalg algebra
layer. Structured objects and optimization spaces add their semantics through
dedicated types, traits, and `coordinate_map` specializations.

`Vector<T, Dim>` follows the same fixed-size ownership model while remaining
generic over its coordinate type. Field-based algorithms require `FieldVector`;
non-field fixed-size data should use `std::array` unless vector semantics are
intended. `VectorContainer` itself promises only registered element and
dimension metadata. `RealVector`, `ComplexVector`, and `CompatibleVectors`
provide focused refinements without implying that every algebraic operation is
available.

Vector construction supports CTAD through the common coordinate type:

```cpp
constexpr auto x = linalg::Vector{1.0, 2.0, 3.0};
static_assert(std::same_as<decltype(x), const linalg::Vector<double, 3>>);
static_assert(x(1) == 2.0);
```

`vector_element_t<Vec>` removes one registered vector layer, whereas
`vector_scalar_t<Vec>` recursively unwraps nested vector containers. Most
linear-algebra algorithms consume the former because they operate on the
immediate field coordinates.

`linalg_ops.hpp` owns arithmetic such as vector/matrix addition, scalar
multiplication, transpose, matrix products, and matrix-vector application.
`linalg_util.hpp` owns structural operations such as half-open slices, block
copy/replacement, triangular extraction, and `svec`/`smat` conversion. Keeping
these operations outside the containers prevents storage types from acquiring
implicit space or operator semantics.

Matrix-generic code reads normalized metadata through `matrix_scalar_t<Mat>`,
`mat_rows_v<Mat>`, and `mat_cols_v<Mat>`. These aliases remove cv/ref
qualification before consulting `mat_traits`, so `Matrixd<2, 3>`,
`const Matrixd<2, 3>`, and `Matrixd<2, 3>&` share one registration. New matrix
representations opt in by specializing `mat_traits` with `value_type`, `ROWS`,
and `COLS`; the unspecialized primary template is intentionally incomplete.

The matrix concepts are focused compile-time classifications:

- `MatrixContainer` requires registered field and shape metadata.
- `SquareMatrix`, `RealMatrix`, and `ComplexMatrix` refine shape or scalar type.
- `MatVecCompatible` checks equal scalar types and matching inner dimensions.
- `MatMulCompatible` checks equal scalar types and compatible matrix shapes.

`MatrixContainer` does not promise arithmetic, element access, symmetry, or
space membership. Algorithms state those additional requirements where they
are used. This keeps a representation trait from silently assigning broader
mathematical meaning to a container.

### Spaces and coordinates

The space layer distinguishes an abstract finite-dimensional space from the C++
type used to represent one of its elements and from the vector used to store its
coordinates. `space_traits<Space>` registers:

- `scalar_type`: the underlying real or complex field;
- `element_type`: the canonical mathematical object representation;
- `coordinate_type`: its finite coordinate representation; and
- `DIM`: the number of independent scalar coordinates.

`Space` checks this metadata. `CoordinatizedSpace` additionally requires an
explicit `coordinate_map<Space>` with exact element-to-coordinate and
coordinate-to-element operations. The free functions make that delegation
visible at call sites:

```cpp
using Space = linalg::MatrixSpace<double, 2, 3>;

constexpr auto coordinate = linalg::coordinates<Space>(matrix);
constexpr auto round_trip = linalg::from_coordinates<Space>(coordinate);
```

`CoordinateSpace<T, Dim>` uses `Vector<T, Dim>` for both representations.
`MatrixSpace<T, Rows, Cols>` flattens matrices in row-major order, mapping
`A(row, col)` to coordinate `row * Cols + col`.

`SymmetricMatrixSpace<T, Dim>` instead has only `Dim * (Dim + 1) / 2`
independent coordinates. Its `svec`/`smat` map applies square-root-of-two
scaling to off-diagonal entries. This makes coordinate inner products agree
with the Frobenius inner product of the represented symmetric matrices. A
`PsdCone<AmbientSpace>` then describes the positive-semidefinite subset by
testing the canonical element's `is_psd()` predicate; PSD membership is not a
property of the plain `Matrix` container.

The older `VectorSpace` contract remains for APIs expressed through
`space_type_t` and `space_point_t`. New representation-aware APIs should prefer
`Space`, `CoordinatizedSpace`, `space_element_t`, and `space_coordinate_t`.
Concepts validate callable expressions and exact result types, but do not prove
the mathematical vector-space axioms or that coordinate conversions are mutual
inverses; implementations and round-trip tests retain that responsibility.

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

See also
[`SOLVER_NUMERICAL_HARDENING.md`](SOLVER_NUMERICAL_HARDENING.md) for solver
conditioning, coefficient-cleanup, and H∞ gain-recovery hardening notes.

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
