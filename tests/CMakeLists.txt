cmake_minimum_required(VERSION 3.22)

include_directories(../core/math/include)
include_directories(../core/physics/include)

if(BUILD_TESTING)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.8.1
  )

  FetchContent_MakeAvailable(Catch2)

  include(CTest)

  enable_testing()

  # Unit Tests
  add_executable(linalg_test unit/linalg_test.cpp)
  add_test(NAME linalg_test COMMAND linalg_test)
  target_link_libraries(linalg_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(linalg_test PRIVATE ../math/include)

  add_executable(linear_systems_test unit/linear_systems_test.cpp)
  add_test(NAME linear_systems_test COMMAND linear_systems_test)
  target_link_libraries(linear_systems_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(linear_systems_test PRIVATE ../math/include)

  add_executable(one_dof_pendulum_test unit/one_dof_pendulum_test.cpp)
  add_test(NAME one_dof_pendulum_test COMMAND one_dof_pendulum_test)
  target_link_libraries(one_dof_pendulum_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(one_dof_pendulum_test PRIVATE ../physics/include)

  add_executable(two_dof_pendulum_test unit/two_dof_pendulum_test.cpp)
  add_test(NAME two_dof_pendulum_test COMMAND two_dof_pendulum_test)
  target_link_libraries(two_dof_pendulum_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(two_dof_pendulum_test PRIVATE ../physics/include)

  add_executable(one_dof_smd_test unit/one_dof_smd_test.cpp)
  add_test(NAME one_dof_smd_test COMMAND one_dof_smd_test)
  target_link_libraries(one_dof_smd_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(one_dof_smd_test PRIVATE ../physics/include)

  add_executable(affine_set_test unit/affine_set_test.cpp)
  add_test(NAME affine_set_test COMMAND affine_set_test)
  target_link_libraries(affine_set_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(affine_set_test PRIVATE ../math/include)

  add_executable(convex_set_test unit/convex_set_test.cpp)
  add_test(NAME convex_set_test COMMAND convex_set_test)
  target_link_libraries(convex_set_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(convex_set_test PRIVATE ../math/include)

  add_executable(convex_fn_test unit/convex_fn_test.cpp)
  add_test(NAME convex_fn_test COMMAND convex_fn_test)
  target_link_libraries(convex_fn_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(convex_fn_test PRIVATE ../math/include)

  add_executable(convex_prob_test unit/convex_prob_test.cpp)
  add_test(NAME convex_prob_test COMMAND convex_prob_test)
  target_link_libraries(convex_prob_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(convex_prob_test PRIVATE ../math/include)

  add_executable(cone_test unit/cone_test.cpp)
  add_test(NAME cone_test COMMAND cone_test)
  target_link_libraries(cone_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(cone_test PRIVATE ../math/include)

  add_executable(kkt_system_test unit/kkt_system_test.cpp)
  add_test(NAME kkt_system_test COMMAND kkt_system_test)
  target_link_libraries(kkt_system_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(kkt_system_test PRIVATE ../math/include)

  add_executable(pd_ipm_test unit/pd_ipm_test.cpp)
  add_test(NAME pd_ipm_test COMMAND pd_ipm_test)
  target_link_libraries(pd_ipm_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(pd_ipm_test PRIVATE ../math/include)

  # Functional Tests
  add_executable(linalg_fuzz_test functional/linalg_fuzz_test.cpp)
  target_include_directories(linalg_fuzz_test PRIVATE ../math/include)

  add_executable(convex_fuzz_test functional/convex_fuzz_test.cpp)
  target_include_directories(convex_fuzz_test PRIVATE ../math/include)
endif()
