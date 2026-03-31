cmake_minimum_required(VERSION 3.22)

include_directories(../core/include)
include_directories(util)

set(UNIT_COMPILE_OPTS
  -ffunction-sections
  -fdata-sections
  -fvisibility=hidden
  -fno-rtti
  -flto
  -Oz
  -DAKREON_DEBUG
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND UNIT_COMPILE_OPTS
    -fconstexpr-depth=65536
    -fconstexpr-steps=2000000000
  )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  list(APPEND UNIT_COMPILE_OPTS
    -fconstexpr-depth=65536
    -fconstexpr-ops-limit=2000000000
  )
endif()

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
  target_include_directories(linalg_test PRIVATE ../include)
  target_compile_options(linalg_test PRIVATE ${UNIT_COMPILE_OPTS} )

  add_executable(linear_systems_test unit/linear_systems_test.cpp)
  add_test(NAME linear_systems_test COMMAND linear_systems_test)
  target_link_libraries(linear_systems_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(linear_systems_test PRIVATE ../include)
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
  target_include_directories(affine_set_test PRIVATE ../include)
  target_compile_options(affine_set_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(convex_set_test unit/convex_set_test.cpp)
  add_test(NAME convex_set_test COMMAND convex_set_test)
  target_link_libraries(convex_set_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(convex_set_test PRIVATE ../include)
  target_compile_options(convex_set_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(convex_fn_test unit/convex_fn_test.cpp)
  add_test(NAME convex_fn_test COMMAND convex_fn_test)
  target_link_libraries(convex_fn_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(convex_fn_test PRIVATE ../include)
  target_compile_options(convex_fn_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(convex_prob_test unit/convex_prob_test.cpp)
  add_test(NAME convex_prob_test COMMAND convex_prob_test)
  target_link_libraries(convex_prob_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(convex_prob_test PRIVATE ../include)
  target_compile_options(convex_prob_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(cone_test unit/cone_test.cpp)
  add_test(NAME cone_test COMMAND cone_test)
  target_link_libraries(cone_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(cone_test PRIVATE ../include)
  target_compile_options(cone_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(lp_test unit/lp_test.cpp)
  add_test(NAME lp_test COMMAND lp_test)
  target_link_libraries(lp_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(lp_test PRIVATE ../include)
  target_compile_options(lp_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(qp_test unit/qp_test.cpp)
  add_test(NAME qp_test COMMAND qp_test)
  target_link_libraries(qp_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(qp_test PRIVATE ../include)
  target_compile_options(qp_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(socp_test unit/socp_test.cpp)
  add_test(NAME socp_test COMMAND socp_test)
  target_link_libraries(socp_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(socp_test PRIVATE ../include)
  target_compile_options(socp_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(sdp_test unit/sdp_test.cpp)
  add_test(NAME sdp_test COMMAND sdp_test)
  target_link_libraries(sdp_test PRIVATE Catch2::Catch2WithMain)
  target_include_directories(sdp_test PRIVATE ../include)
  target_compile_options(sdp_test PRIVATE ${UNIT_COMPILE_OPTS})
endif()

if(BUILD_FUNCTIONAL_TESTS)
  add_executable(linalg_fuzz_test functional/linalg_fuzz_test.cpp)
  target_include_directories(linalg_fuzz_test PRIVATE ../include)
  target_compile_options(linalg_fuzz_test PRIVATE ${UNIT_COMPILE_OPTS})

  add_executable(lp_fuzz_test functional/lp_fuzz_test.cpp)
  target_include_directories(lp_fuzz_test PRIVATE ../include)
  target_compile_options(lp_fuzz_test PRIVATE ${FUNC_COMPILE_OPTS})

  add_executable(qp_fuzz_test functional/qp_fuzz_test.cpp)
  target_include_directories(qp_fuzz_test PRIVATE ../include)
  target_compile_options(qp_fuzz_test PRIVATE ${FUNC_COMPILE_OPTS})

  add_executable(sdp_fuzz_test functional/sdp_fuzz_test.cpp)
  target_include_directories(sdp_fuzz_test PRIVATE ../include)
  target_compile_options(sdp_fuzz_test PRIVATE ${FUNC_COMPILE_OPTS})
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
  lp_test
  qp_test
  socp_test
  linalg_fuzz_test
  lp_fuzz_test
  qp_fuzz_test
  sdp_fuzz_test
)

foreach(_ak_target IN LISTS AKREON_TEST_TARGETS)
  if(TARGET ${_ak_target})
    akreon_configure_target(${_ak_target})
  endif()
endforeach()
