# toolchain-arm-none-eabi.cmake

cmake_minimum_required(VERSION 3.22)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Don't try to run test binaries on host
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Toolchain prefix
set(TOOLCHAIN_PREFIX arm-none-eabi)
set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++)
set(CMAKE_ASM_COMPILER ${TOOLCHAIN_PREFIX}-as)
set(CMAKE_AR           ${TOOLCHAIN_PREFIX}-ar)
set(CMAKE_OBJCOPY      ${TOOLCHAIN_PREFIX}-objcopy)
set(CMAKE_OBJDUMP      ${TOOLCHAIN_PREFIX}-objdump)
set(CMAKE_SIZE         ${TOOLCHAIN_PREFIX}-size)

# CPU / FPU / Thumb
set(MCPU_FLAGS  "-mcpu=cortex-m4")
set(MFPU_FLAGS  "-mfpu=fpv4-sp-d16 -mfloat-abi=hard")
set(MTHUMB_FLAG "-mthumb")

set(COMMON_FLAGS "${MCPU_FLAGS} ${MFPU_FLAGS} ${MTHUMB_FLAG} -ffunction-sections -fdata-sections")

# Seed language flags
set(CMAKE_C_FLAGS_INIT   "${COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${COMMON_FLAGS} -fno-exceptions -fno-rtti")

# Linker flags: newlib-nano + gc-sections.
# nosys/syscall behavior is configured at project level via AKREON_USE_NOSYS.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-Wl,--gc-sections --specs=nano.specs")

# Device macro (for HAL/CMSIS)
add_compile_definitions(STM32F407xx)
