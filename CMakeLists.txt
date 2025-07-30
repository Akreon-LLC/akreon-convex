cmake_minimum_required(VERSION 3.22)

project(dmc VERSION 0.0.0 LANGUAGES CXX DESCRIPTION "Akreon SDK" HOMEPAGE_URL https://github.com/dakre21/akreon_sdk)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(BUILD_TESTING "Enable building test code" ON)

include(FetchContent)

add_subdirectory(math)

if(BUILD_TESTING)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.8.1
  )
  
  FetchContent_MakeAvailable(Catch2)
  
  include(CTest)

  enable_testing()

  add_executable(algebra_test tests/algebra_test.cpp)
  add_test(NAME algebra_test COMMAND algebra_test)
  target_link_libraries(algebra_test PRIVATE Catch2::Catch2WithMain Eigen3::Eigen math)
  target_include_directories(algebra_test PRIVATE ../math/include)
  
  add_executable(topology_test tests/topology_test.cpp)
  add_test(NAME topology_test COMMAND topology_test)
  target_link_libraries(topology_test PRIVATE Catch2::Catch2WithMain Eigen3::Eigen math)
  target_include_directories(topology_test PRIVATE ../math/include)

  #add_executable(manifold_test tests/manifold_test.cpp)
  #add_test(NAME manifold_test COMMAND manifold_test)
  #target_link_libraries(manifold_test PRIVATE Catch2::Catch2WithMain Eigen3::Eigen math)
  #target_include_directories(manifold_test PRIVATE ../math/include)
endif()
