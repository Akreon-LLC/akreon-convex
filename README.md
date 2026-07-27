# akreon-sdk

`akreon-sdk` is the constexpr-oriented C++ foundation for Akreon's fixed-size
linear algebra, convex optimization, dynamics, and control synthesis APIs.
Dimensions and problem structure are expressed through compile-time types.

`akreon-native` is the primary runtime integration: it includes the SDK headers,
compiles manifest-selected control problem shapes into native code, and exposes
those static SDK specializations through a stable C ABI and Python bindings.

## Legal

- Proprietary license: `LICENSE`
- IP protection playbook: `LEGAL.md`
- Contribution policy: `CONTRIBUTING.md`
- NDA template: `legal/templates/NDA.md`
- Evaluation order form template: `legal/templates/Evaluation-Order-Form.md`
- Contributor agreement template: `legal/Contributor-Agreement.md`

## SDK Development

**Recommended: CMake Presets**

1. `cmake --preset host-default`
1. `cmake --build --preset host-default -j`

Available local developer/test presets:

- `host-default`
- `host-functional-only`

Notes:

- `host-default` requires network access for `Catch2` via `FetchContent`.
- Use `host-functional-only` in offline environments.

This repository's CMake surface is for SDK development and verification. The
SDK does not currently publish a CMake library target; C++ consumers add
`core/include` to their include path and include the relevant umbrella header:

- `math.hpp` for linear algebra and convex optimization
- `control.hpp` for control models and synthesis
- `dynamics.hpp` for dynamical-system abstractions

**Manual flags (if needed)**

1. `cmake -S . -B <build-dir> -DBUILD_UNIT_TESTS=<ON|OFF> -DBUILD_FUNCTIONAL_TESTS=<ON|OFF>`
1. `cmake --build <build-dir> -j`

## Integration Through akreon-native

The sibling `akreon-native` repository is the supported runtime and Python
integration layer. By default, its CMake configuration resolves the SDK from
`../akreon-sdk/core/include`. A different checkout can be selected explicitly:

```bash
cmake -S ../akreon-native -B ../akreon-native/build \
  -DAKREON_SDK_INCLUDE_DIR=/path/to/akreon-sdk/core/include
cmake --build ../akreon-native/build -j
```

Native control manifests select the dimensions and problem families compiled
into the binary. `akreon-native` then delegates to the corresponding SDK
templates behind its C ABI and nanobind Python module. Runtime users consume
`akreon-native` or a generated customer wheel; they do not need to configure
or build the SDK directly.

## Test

1. `ctest --test-dir out/build/host-default/tests --output-on-failure` (unit tests)
1. `./out/build/host-default/tests/linalg_fuzz_test` for randomized linalg checks

See [docs/CODE_REFERENCE.md](docs/CODE_REFERENCE.md) for the header layout,
compile-time model, and native integration boundary.
