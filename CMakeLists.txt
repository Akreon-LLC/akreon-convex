cmake_minimum_required(VERSION 3.22)

project(akreon VERSION 0.0.0 LANGUAGES CXX DESCRIPTION "Akreon SDK" HOMEPAGE_URL https://github.com/dakre21/akreon_sdk)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Werror)
endif()

option(BUILD_UNIT_TESTS "Build Catch2-based unit tests" ON)
option(BUILD_FUNCTIONAL_TESTS "Build functional/fuzz test executables" ON)
set(AKREON_PLATFORM "baremetal" CACHE STRING
    "Target runtime platform: baremetal or freertos")
set_property(CACHE AKREON_PLATFORM PROPERTY STRINGS baremetal freertos)
option(AKREON_USE_NOSYS
       "Link bare-metal executables with --specs=nosys.specs syscall stubs" ON)

if(NOT AKREON_PLATFORM STREQUAL "baremetal" AND
   NOT AKREON_PLATFORM STREQUAL "freertos")
  message(FATAL_ERROR
          "AKREON_PLATFORM must be one of: baremetal, freertos")
endif()

if(AKREON_PLATFORM STREQUAL "freertos" AND AKREON_USE_NOSYS)
  message(WARNING
          "AKREON_USE_NOSYS is ON while AKREON_PLATFORM=freertos. "
          "FreeRTOS applications typically provide syscall retargets.")
endif()

function(akreon_configure_target target_name)
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "akreon_configure_target: target '${target_name}' not found")
  endif()

  if(AKREON_PLATFORM STREQUAL "baremetal")
    target_compile_definitions(${target_name} PRIVATE AKREON_PLATFORM_BAREMETAL=1)
  elseif(AKREON_PLATFORM STREQUAL "freertos")
    target_compile_definitions(${target_name} PRIVATE AKREON_PLATFORM_FREERTOS=1)
  endif()

  if(AKREON_USE_NOSYS)
    target_compile_definitions(${target_name} PRIVATE AKREON_USE_NOSYS=1)
  else()
    target_compile_definitions(${target_name} PRIVATE AKREON_USE_NOSYS=0)
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "Generic" AND NOT AKREON_USE_NOSYS)
    message(STATUS
            "AKREON_USE_NOSYS=OFF: '${target_name}' expects user/application "
            "syscall retarget implementations (_write, _read, _sbrk, etc).")
  endif()
endfunction()

include(FetchContent)

add_subdirectory(tests)
