# akreon_sdk

## Build
`AKREON_PLATFORM`:
- `baremetal` (default)
- `freertos`

`AKREON_USE_NOSYS`:
- `ON` (default): link bare-metal executables with `--specs=nosys.specs`
- `OFF`: expect user/app syscall retargets (`_write`, `_read`, `_sbrk`, ...)

**Recommended: CMake Presets**
1. `cmake --preset host-default`
1. `cmake --build --preset host-default -j`

Available configure/build presets:
- `host-default`
- `host-functional-only`
- `stm32-baremetal-nosys`
- `stm32-freertos`

Notes:
- `host-default` requires network access for `Catch2` via `FetchContent`.
- Use `host-functional-only` in offline environments.

**STM32 functional targets (preset examples)**
1. `cmake --preset stm32-baremetal-nosys`
1. `cmake --build --preset stm32-baremetal-nosys --target convex_fuzz_test -j`
1. `cmake --build --preset stm32-baremetal-nosys --target linalg_fuzz_test -j`

**Manual flags (if needed)**
1. `cmake -S . -B <build-dir> -DCMAKE_TOOLCHAIN_FILE=<...> -DBUILD_UNIT_TESTS=<ON|OFF> -DBUILD_FUNCTIONAL_TESTS=<ON|OFF> -DAKREON_PLATFORM=<baremetal|freertos> -DAKREON_USE_NOSYS=<ON|OFF>`
1. `cmake --build <build-dir> -j`

**STM32/FreeRTOS style config (app provides syscalls)**
1. `cmake --preset stm32-freertos`
1. `cmake --build --preset stm32-freertos --target convex_fuzz_test -j`

## Test
1. `cd build && ctest --output-on-failure` (unit tests only)
1. `./build/tests/linalg_fuzz_test` to run randomized linalg functional checks
1. `./build/tests/convex_fuzz_test` to run randomized convex functional checks
