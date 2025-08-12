cmake_minimum_required(VERSION 3.22)

# Example toolchain.cmake snippet
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Prevent CMake from trying to run test binaries on the host
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Specify the path to your ARM GCC toolchain
set(TOOLCHAIN_PREFIX arm-none-eabi)
set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++)
set(CMAKE_ASM_COMPILER ${TOOLCHAIN_PREFIX}-as)
set(CMAKE_AR           ${TOOLCHAIN_PREFIX}-ar)
set(CMAKE_OBJCOPY      ${TOOLCHAIN_PREFIX}-objcopy)
set(CMAKE_OBJDUMP      ${TOOLCHAIN_PREFIX}-objdump)
set(CMAKE_SIZE         ${TOOLCHAIN_PREFIX}-size)

# --- Target selection (EDIT THESE) ---
# Example: STM32F4 (Cortex-M4F: FPU=fpv4-sp-d16, hard-float ABI)
set(MCPU_FLAGS "-mcpu=cortex-m4")
set(MFPU_FLAGS "-mfpu=fpv4-sp-d16 -mfloat-abi=hard")
set(MTHUMB_FLAG "-mthumb")
# Define your device macro (used by HAL/CMSIS)
add_compile_definitions(STM32F407xx)
