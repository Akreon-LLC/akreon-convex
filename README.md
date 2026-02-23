# akreon_sdk

## Legal
- Proprietary license: `LICENSE`
- IP protection playbook: `LEGAL.md`
- Contribution policy: `CONTRIBUTING.md`
- NDA template: `legal/templates/NDA.md`
- Evaluation order form template: `legal/templates/Evaluation-Order-Form.md`
- Contributor agreement template: `legal/Contributor-Agreement.md`

## Build
Cross toolchain file:
- `toolchains/arm-none-eabi-stm32.cmake`

Primary STM32 cache variables (set by presets):
- `AKREON_CPU`
- `AKREON_FPU`
- `AKREON_FLOAT_ABI`
- `AKREON_THUMB`
- `AKREON_OPT_LEVEL`
- `AKREON_USE_LTO`
- `AKREON_DEVICE_DEFS`
- `AKREON_USE_NOSYS`

**Recommended: CMake Presets**
1. `cmake --preset host-default`
1. `cmake --build --preset host-default -j`

Available configure/build presets:
- `host-default`
- `host-functional-only`
- `crazyflie` (STM32F405 / Cortex-M4F)
- `pixhawk` (STM32F7 / Cortex-M7F default)
- `cubeorange` (alias of `pixhawk`)

Notes:
- `host-default` requires network access for `Catch2` via `FetchContent`.
- Use `host-functional-only` in offline environments.

**Crazyflie (STM32F405)**
1. `cmake --preset crazyflie`
1. `cmake --build --preset crazyflie -j`

**Pixhawk (STM32F7 default)**
1. `cmake --preset pixhawk`
1. `cmake --build --preset pixhawk -j`

**CubeOrange alias**
1. `cmake --preset cubeorange`
1. `cmake --build --preset cubeorange -j`

**Build specific functional targets**
1. `cmake --build --preset crazyflie --target convex_fuzz_test -j`
1. `cmake --build --preset crazyflie --target linalg_fuzz_test -j`

**Manual flags (if needed)**
1. `cmake -S . -B <build-dir> -DCMAKE_TOOLCHAIN_FILE=toolchains/arm-none-eabi-stm32.cmake -DBUILD_UNIT_TESTS=<ON|OFF> -DBUILD_FUNCTIONAL_TESTS=<ON|OFF> -DAKREON_CPU=<...> -DAKREON_FPU=<...> -DAKREON_FLOAT_ABI=<hard|softfp|soft> -DAKREON_DEVICE_DEFS=<DEF1;DEF2> -DAKREON_OPT_LEVEL=<...> -DAKREON_USE_LTO=<ON|OFF> -DAKREON_USE_NOSYS=<ON|OFF>`
1. `cmake --build <build-dir> -j`

## Test
1. `ctest --test-dir out/build/host-default/tests --output-on-failure` (unit tests)
1. `./out/build/host-default/tests/linalg_fuzz_test` for randomized linalg checks
1. `./out/build/host-default/tests/convex_fuzz_test` for randomized convex checks
