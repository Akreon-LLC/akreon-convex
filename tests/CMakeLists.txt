cmake_minimum_required(VERSION 3.22)

include_directories(../core/math/include)
include_directories(../core/physics/include)

set(UNIT_COMPILE_OPTS
  -ffunction-sections
  -fdata-sections
  -fvisibility=hidden
  -fno-rtti
  -flto
  -Oz
  -DAKREON_DEBUG
)
set(FUNC_COMPILE_OPTS
  -ffunction-sections
  -fdata-sections
  -fvisibility=hidden
  -fno-rtti
  -flto
  -fno-exceptions
  -fno-unwind-tables
  -fno-asynchronous-unwind-tables
  -Oz
  -DAKREON_DEBUG
)

if(BUILD_UNIT_TESTS)
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
  target_compile_options(linalg_test PRIVATE ${UNIT_COMPILE_OPTS} )

  add_executable(linear_systems_test unit/linear_systems_test.cpp)
  add_test(NAME linear_systems_test COMMAND linear_systems_test)
  target_link_libraries(linear_systems_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(linear_systems_test PRIVATE ../math/include)
  target_compile_options(linear_systems_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(one_dof_pendulum_test unit/one_dof_pendulum_test.cpp)
  add_test(NAME one_dof_pendulum_test COMMAND one_dof_pendulum_test)
  target_link_libraries(one_dof_pendulum_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(one_dof_pendulum_test PRIVATE ../physics/include)
  target_compile_options(one_dof_pendulum_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(two_dof_pendulum_test unit/two_dof_pendulum_test.cpp)
  add_test(NAME two_dof_pendulum_test COMMAND two_dof_pendulum_test)
  target_link_libraries(two_dof_pendulum_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(two_dof_pendulum_test PRIVATE ../physics/include)
  target_compile_options(two_dof_pendulum_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(one_dof_smd_test unit/one_dof_smd_test.cpp)
  add_test(NAME one_dof_smd_test COMMAND one_dof_smd_test)
  target_link_libraries(one_dof_smd_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(one_dof_smd_test PRIVATE ../physics/include)
  target_compile_options(one_dof_smd_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(affine_set_test unit/affine_set_test.cpp)
  add_test(NAME affine_set_test COMMAND affine_set_test)
  target_link_libraries(affine_set_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(affine_set_test PRIVATE ../math/include)
  target_compile_options(affine_set_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(convex_set_test unit/convex_set_test.cpp)
  add_test(NAME convex_set_test COMMAND convex_set_test)
  target_link_libraries(convex_set_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(convex_set_test PRIVATE ../math/include)
  target_compile_options(convex_set_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(convex_fn_test unit/convex_fn_test.cpp)
  add_test(NAME convex_fn_test COMMAND convex_fn_test)
  target_link_libraries(convex_fn_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(convex_fn_test PRIVATE ../math/include)
  target_compile_options(convex_fn_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(convex_prob_test unit/convex_prob_test.cpp)
  add_test(NAME convex_prob_test COMMAND convex_prob_test)
  target_link_libraries(convex_prob_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(convex_prob_test PRIVATE ../math/include)
  target_compile_options(convex_prob_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(cone_test unit/cone_test.cpp)
  add_test(NAME cone_test COMMAND cone_test)
  target_link_libraries(cone_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(cone_test PRIVATE ../math/include)
  target_compile_options(cone_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(pd_ipm_test unit/pd_ipm_test.cpp)
  add_test(NAME pd_ipm_test COMMAND pd_ipm_test)
  target_link_libraries(pd_ipm_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(pd_ipm_test PRIVATE ../math/include)
  target_compile_options(pd_ipm_test PRIVATE ${UNIT_COMPILE_OPTS})
endif()

if(BUILD_FUNCTIONAL_TESTS)
  add_executable(linalg_fuzz_test functional/linalg_fuzz_test.cpp)
  target_include_directories(linalg_fuzz_test PRIVATE ../math/include)
  target_compile_options(linalg_fuzz_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(convex_fuzz_test functional/convex_fuzz_test.cpp)
  target_include_directories(convex_fuzz_test PRIVATE ../math/include)
  target_compile_options(convex_fuzz_test PRIVATE ${FUNC_COMPILE_OPTS})
endif()

set(AKREON_TEST_TARGETS
  linalg_test
  linear_systems_test
  one_dof_pendulum_test
  two_dof_pendulum_test
  one_dof_smd_test
  affine_set_test
  convex_set_test
  convex_fn_test
  convex_prob_test
  cone_test
  pd_ipm_test
  linalg_fuzz_test
  convex_fuzz_test
)

foreach(_ak_target IN LISTS AKREON_TEST_TARGETS)
  if(TARGET ${_ak_target})
    akreon_configure_target(${_ak_target})
  endif()
endforeach()
