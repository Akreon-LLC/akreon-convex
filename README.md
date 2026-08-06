# akreon-core

`akreon-core` is the constexpr-oriented C++ foundation for Akreon's fixed size
linear algebra, convex optimization, dynamics, and control synthesis APIs.
Dimensions and problem structure are expressed through compile time types.

`akreon-python` is the primary runtime integration: it includes the core headers,
compiles manifest selected control problem shapes into native code, and exposes
those static core specializations through a stable C ABI and Python bindings.

## Legal

- Proprietary license: `LICENSE`
- IP protection playbook: `LEGAL.md`
- Contribution policy: `CONTRIBUTING.md`
- NDA template: `legal/templates/NDA.md`
- Evaluation order form template: `legal/templates/Evaluation-Order-Form.md`
- Contributor agreement template: `legal/Contributor-Agreement.md`

## Core Development

**Recommended: CMake Presets**

1. `cmake --preset host-default`
1. `cmake --build --preset host-default -j`

Available local developer/test presets:

- `host-default`
- `host-functional-only`

Notes:

- `host-default` requires network access for `Catch2` via `FetchContent`.
- Use `host-functional-only` in offline environments.

This repository's CMake surface is for core development and verification. The
core does not currently publish a CMake library target; C++ consumers add
`core/include` to their include path and include the relevant umbrella header:

- `math.hpp` for linear algebra and convex optimization
- `control.hpp` for control models and synthesis
- `dynamics.hpp` for dynamical-system abstractions

The linalg API uses one field-aware vector pairing:

```cpp
constexpr linalg::Vector<double, 3> x{1.0, 2.0, 3.0};
constexpr linalg::Vector<double, 3> y{4.0, 5.0, 6.0};
constexpr auto xy = linalg::inner_product(x, y);
```

`inner_product(x, y)` delegates to `StandardInnerProduct`. Over real fields it
computes `sum(x(i) * y(i))`; over complex fields it computes
`sum(conj(x(i)) * y(i))`. Norm functions expose explicit domain, scalar, and
real codomain associated types through `norm_traits`.

**Manual flags (if needed)**

1. `cmake -S . -B <build-dir> -DBUILD_UNIT_TESTS=<ON|OFF> -DBUILD_FUNCTIONAL_TESTS=<ON|OFF>`
1. `cmake --build <build-dir> -j`

## Integration Through akreon-python

The sibling `akreon-python` repository is the supported runtime and Python
integration layer. By default, its CMake configuration resolves the core from
`../akreon-core/core/include`. A different checkout can be selected explicitly:

```bash
cmake -S ../akreon-python -B ../akreon-python/build \
  -DAKREON_CORE_INCLUDE_DIR=/path/to/akreon-core/core/include
cmake --build ../akreon-python/build -j
```

Control manifests select the dimensions and problem families compiled into the
binary. `akreon-python` then delegates to the corresponding core templates
behind its C ABI and nanobind Python module. Runtime users consume
`akreon-python` or a generated customer wheel; they do not need to configure
or build the core directly.

## Test

1. `ctest --test-dir out/build/host-default/tests --output-on-failure` (unit tests)
1. `./out/build/host-default/tests/linalg_fuzz_test` for randomized linalg checks

See [docs/CODE_REFERENCE.md](docs/CODE_REFERENCE.md) for the header layout,
compile-time model, and Python integration boundary.
