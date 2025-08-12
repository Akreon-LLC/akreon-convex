# akreon_sdk

## Build
**On Host**
1. ```cmake -S . -B build```
1. ```cmake --build build -j```

**Cross Compile STM32**
1. ```cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi-stm32.cmake -DBUILD_TESTING=OFF```
1. ```cmake --build build -j```

## Test
1. ```cd build && ctest```
