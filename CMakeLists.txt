cmake_minimum_required(VERSION 3.22)

project(akreon VERSION 0.0.0 LANGUAGES CXX DESCRIPTION "Akreon SDK" HOMEPAGE_URL https://github.com/dakre21/akreon_sdk)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Werror)
endif()

option(BUILD_TESTING "Build testing code" ON)
option(BUILD_CORE "Build core library" ON)
option(BUILD_UI "Build ui app" ON)

include(FetchContent)

add_subdirectory(core)
add_subdirectory(ui)
add_subdirectory(tests)
