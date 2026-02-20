set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(TOOLCHAIN_PREFIX arm-none-eabi CACHE STRING "GNU Arm Embedded toolchain prefix")

set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++)
set(CMAKE_ASM_COMPILER ${TOOLCHAIN_PREFIX}-as)
set(CMAKE_AR           ${TOOLCHAIN_PREFIX}-ar)
set(CMAKE_OBJCOPY      ${TOOLCHAIN_PREFIX}-objcopy)
set(CMAKE_OBJDUMP      ${TOOLCHAIN_PREFIX}-objdump)
set(CMAKE_SIZE         ${TOOLCHAIN_PREFIX}-size)

set(AKREON_CPU cortex-m4 CACHE STRING "ARM CPU core (e.g. cortex-m4, cortex-m7, cortex-m33)")
set(AKREON_FPU fpv4-sp-d16 CACHE STRING "ARM FPU unit (e.g. fpv4-sp-d16, fpv5-d16)")
set(AKREON_FLOAT_ABI hard CACHE STRING "ARM float ABI (hard, softfp, soft)")
set_property(CACHE AKREON_FLOAT_ABI PROPERTY STRINGS hard softfp soft)
set(AKREON_THUMB ON CACHE BOOL "Build Thumb ISA (-mthumb)")
set(AKREON_OPT_LEVEL -Os CACHE STRING "Optimization level (e.g. -Os, -O2, -Og)")
set(AKREON_USE_LTO ON CACHE BOOL "Enable link-time optimization")
set(AKREON_DEVICE_DEFS "" CACHE STRING "Semicolon-separated device macros (e.g. STM32F405xx;USE_HAL_DRIVER)")
set(AKREON_USE_NOSYS ON CACHE BOOL "Link with --specs=nosys.specs")

set(_ak_cpu_flags
  -mcpu=${AKREON_CPU}
  -mfloat-abi=${AKREON_FLOAT_ABI}
  -ffunction-sections
  -fdata-sections
  -fno-unwind-tables
  -fno-asynchronous-unwind-tables
  ${AKREON_OPT_LEVEL}
)

if(NOT AKREON_FPU STREQUAL "")
  list(APPEND _ak_cpu_flags -mfpu=${AKREON_FPU})
endif()

if(AKREON_THUMB)
  list(APPEND _ak_cpu_flags -mthumb)
endif()

foreach(_ak_def IN LISTS AKREON_DEVICE_DEFS)
  if(NOT _ak_def STREQUAL "")
    list(APPEND _ak_cpu_flags -D${_ak_def})
  endif()
endforeach()

set(_ak_c_flags ${_ak_cpu_flags})
set(_ak_cxx_flags ${_ak_cpu_flags} -fno-exceptions -fno-rtti)
set(_ak_asm_flags -mcpu=${AKREON_CPU})

if(NOT AKREON_FPU STREQUAL "")
  list(APPEND _ak_asm_flags -mfpu=${AKREON_FPU})
endif()

if(AKREON_THUMB)
  list(APPEND _ak_asm_flags -mthumb)
endif()

set(_ak_linker_flags
  -mcpu=${AKREON_CPU}
  -mfloat-abi=${AKREON_FLOAT_ABI}
  -Wl,--gc-sections
  --specs=nano.specs
)

if(NOT AKREON_FPU STREQUAL "")
  list(APPEND _ak_linker_flags -mfpu=${AKREON_FPU})
endif()

if(AKREON_THUMB)
  list(APPEND _ak_linker_flags -mthumb)
endif()

if(AKREON_USE_NOSYS)
  list(APPEND _ak_linker_flags --specs=nosys.specs)
endif()

if(AKREON_USE_LTO)
  list(APPEND _ak_c_flags -flto)
  list(APPEND _ak_cxx_flags -flto)
  list(APPEND _ak_linker_flags -flto)
endif()

string(JOIN " " CMAKE_C_FLAGS_INIT ${_ak_c_flags})
string(JOIN " " CMAKE_CXX_FLAGS_INIT ${_ak_cxx_flags})
string(JOIN " " CMAKE_ASM_FLAGS_INIT ${_ak_asm_flags})
string(JOIN " " CMAKE_EXE_LINKER_FLAGS_INIT ${_ak_linker_flags})
