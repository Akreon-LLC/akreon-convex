cmake_minimum_required(VERSION 3.22)

project(dmc VERSION 0.0.0 LANGUAGES CXX DESCRIPTION "Akreon SDK" HOMEPAGE_URL https://github.com/dakre21/akreon_sdk)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(BUILD_TESTING "Enable building test code" ON)

include(FetchContent)

include_directories(
  math/include
  physics/include
)

if(BUILD_TESTING)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.8.1
  )

  FetchContent_MakeAvailable(Catch2)

  include(CTest)

  enable_testing()

  add_executable(algebra_test tests/unit/algebra_test.cpp)
  add_test(NAME algebra_test COMMAND algebra_test)
  target_link_libraries(algebra_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(algebra_test PRIVATE ../math/include)

  add_executable(topology_test tests/unit/topology_test.cpp)
  add_test(NAME topology_test COMMAND topology_test)
  target_link_libraries(topology_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(topology_test PRIVATE ../math/include)

  add_executable(linear_systems_test tests/unit/linear_systems_test.cpp)
  add_test(NAME linear_systems_test COMMAND linear_systems_test)
  target_link_libraries(linear_systems_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(linear_systems_test PRIVATE ../math/include)

  add_executable(linalg_fuzz_test tests/functional/linalg_fuzz_test.cpp)
  target_include_directories(linalg_fuzz_test PRIVATE ../math/include)

  add_executable(one_dof_pendulum_test tests/unit/one_dof_pendulum_test.cpp)
  add_test(NAME one_dof_pendulum_test COMMAND one_dof_pendulum_test)
  target_link_libraries(one_dof_pendulum_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(one_dof_pendulum_test PRIVATE ../physics/include)

  add_executable(two_dof_pendulum_test tests/unit/two_dof_pendulum_test.cpp)
  add_test(NAME two_dof_pendulum_test COMMAND two_dof_pendulum_test)
  target_link_libraries(two_dof_pendulum_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(two_dof_pendulum_test PRIVATE ../physics/include)
endif()
