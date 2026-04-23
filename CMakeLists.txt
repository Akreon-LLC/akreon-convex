cmake_minimum_required(VERSION 3.22)

project(akreon VERSION 0.0.0 LANGUAGES CXX DESCRIPTION "Akreon SDK" HOMEPAGE_URL https://github.com/dakre21/akreon_sdk)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(AKREON_CLANG_FORMAT_MAJOR 22)

find_program(CLANG_FORMAT_BIN NAMES clang-format REQUIRED)

execute_process(
  COMMAND ${CLANG_FORMAT_BIN} --version
  OUTPUT_VARIABLE CLANG_FORMAT_VERSION_OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(
  REGEX MATCH
  "clang-format version ([0-9]+)"
  CLANG_FORMAT_VERSION_MATCH
  "${CLANG_FORMAT_VERSION_OUTPUT}"
)

if(NOT CLANG_FORMAT_VERSION_MATCH)
  message(
    FATAL_ERROR
      "Unable to determine clang-format version from: "
      "${CLANG_FORMAT_VERSION_OUTPUT}"
  )
endif()

set(CLANG_FORMAT_VERSION_MAJOR "${CMAKE_MATCH_1}")

if(NOT CLANG_FORMAT_VERSION_MAJOR STREQUAL "${AKREON_CLANG_FORMAT_MAJOR}")
  message(
    FATAL_ERROR
      "akreon-sdk requires clang-format ${AKREON_CLANG_FORMAT_MAJOR}.x, "
      "but found: ${CLANG_FORMAT_VERSION_OUTPUT}"
  )
endif()

file(
  GLOB_RECURSE AKREON_FORMAT_FILES
  CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/core/include/*.hpp"
  "${CMAKE_SOURCE_DIR}/tests/*.hpp"
  "${CMAKE_SOURCE_DIR}/tests/*.cpp"
)

if(CLANG_FORMAT_BIN)
  add_custom_target(
    clang_format_check
    COMMAND ${CLANG_FORMAT_BIN} --dry-run --Werror ${AKREON_FORMAT_FILES}
    COMMENT "Checking C/C++ formatting with clang-format"
    VERBATIM
  )

  add_custom_target(
    clang_format_fix
    COMMAND ${CLANG_FORMAT_BIN} -i ${AKREON_FORMAT_FILES}
    COMMENT "Formatting C/C++ sources with clang-format"
    VERBATIM
  )
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Werror)
endif()

option(BUILD_UNIT_TESTS "Build Catch2-based unit tests" ON)
option(BUILD_FUNCTIONAL_TESTS "Build functional/fuzz test executables" ON)

function(akreon_configure_target target_name)
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "akreon_configure_target: target '${target_name}' not found")
  endif()
endfunction()

include(FetchContent)

add_subdirectory(tests)
