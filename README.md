# akreon_sdk

## Legal
- Proprietary license: `LICENSE`
- IP protection playbook: `LEGAL.md`
- Contribution policy: `CONTRIBUTING.md`
- NDA template: `tools/legal/templates/NDA.md`
- Evaluation order form template: `tools/legal/templates/Evaluation-Order-Form.md`
- Contributor agreement template: `tools/legal/Contributor-Agreement.md`

## Build
**Recommended: CMake Presets**
1. `cmake --preset host-default`
1. `cmake --build --preset host-default -j`

Available local developer/test presets:
- `host-default`
- `host-functional-only`

Notes:
- `host-default` requires network access for `Catch2` via `FetchContent`.
- Use `host-functional-only` in offline environments.

This repo's CMake surface is for local SDK development and verification. Production deploy-target cross-compilation is expected to live in `akreon-runner`, which can consume this repo's headers and generated sources there.

**Manual flags (if needed)**
1. `cmake -S . -B <build-dir> -DBUILD_UNIT_TESTS=<ON|OFF> -DBUILD_FUNCTIONAL_TESTS=<ON|OFF>`
1. `cmake --build <build-dir> -j`

## Test
1. `ctest --test-dir out/build/host-default/tests --output-on-failure` (unit tests)
1. `./out/build/host-default/tests/linalg_fuzz_test` for randomized linalg checks
1. `./out/build/host-default/tests/convex_fuzz_test` for randomized convex checks
