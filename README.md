# akreon_sdk

## Build
**On Host**
1. cmake -S . -B build
1. cmake --build build -j

**Cross Compile STM32**
1. cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi-stm32.cmake 
1. cmake --build build -j
