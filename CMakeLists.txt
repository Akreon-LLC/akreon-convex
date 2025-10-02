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

include_directories(
  math/include
  physics/include
  controls/include
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

  add_executable(linalg_test tests/unit/linalg_test.cpp)
  add_test(NAME linalg_test COMMAND linalg_test)
  target_link_libraries(linalg_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(linalg_test PRIVATE ../math/include)

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

  add_executable(one_dof_smd_test tests/unit/one_dof_smd_test.cpp)
  add_test(NAME one_dof_smd_test COMMAND one_dof_smd_test)
  target_link_libraries(one_dof_smd_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(one_dof_smd_test PRIVATE ../physics/include)

  add_executable(lti_systems_test tests/unit/lti_systems_test.cpp)
  add_test(NAME lti_systems_test COMMAND lti_systems_test)
  target_link_libraries(lti_systems_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(lti_systems_test PRIVATE ../controls/include)

  add_executable(affine_set_test tests/unit/affine_set_test.cpp)
  add_test(NAME affine_set_test COMMAND affine_set_test)
  target_link_libraries(affine_set_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(affine_set_test PRIVATE ../math/include)

  add_executable(convex_set_test tests/unit/convex_set_test.cpp)
  add_test(NAME convex_set_test COMMAND convex_set_test)
  target_link_libraries(convex_set_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(convex_set_test PRIVATE ../math/include)
endif()
