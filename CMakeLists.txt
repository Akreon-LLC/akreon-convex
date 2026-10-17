cmake_minimum_required(VERSION 3.30)
project(AkreonConvex VERSION 0.1.0 LANGUAGES CXX)
option(BUILD_UNIT_TESTS "Build Catch2 unit tests" ${PROJECT_IS_TOP_LEVEL})
option(BUILD_FUNCTIONAL_TESTS "Build functional tests" ${PROJECT_IS_TOP_LEVEL})
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(cmake/AkreonDependencies.cmake)
akreon_resolve_dependency(linalg AkreonLinalg Akreon::Linalg)

add_library(akreon_convex INTERFACE)
add_library(Akreon::Convex ALIAS akreon_convex)
set_target_properties(akreon_convex PROPERTIES EXPORT_NAME Convex)
target_compile_features(akreon_convex INTERFACE cxx_std_26)
target_include_directories(akreon_convex INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(akreon_convex INTERFACE Akreon::Linalg)
add_library(akreon_convex_test_support INTERFACE)
add_library(Akreon::ConvexTestSupport ALIAS akreon_convex_test_support)
set_target_properties(akreon_convex_test_support PROPERTIES EXPORT_NAME ConvexTestSupport)
target_link_libraries(akreon_convex_test_support INTERFACE Akreon::Convex Akreon::LinalgTestSupport)
target_include_directories(akreon_convex_test_support INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test-support>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_DATADIR}/akreon-convex/test-support>)
install(DIRECTORY test-support/ DESTINATION ${CMAKE_INSTALL_DATADIR}/akreon-convex/test-support)

install(TARGETS akreon_convex akreon_convex_test_support EXPORT AkreonConvexTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT AkreonConvexTargets NAMESPACE Akreon:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AkreonConvex)
configure_package_config_file(cmake/AkreonConvexConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/AkreonConvexConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AkreonConvex)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/AkreonConvexConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/AkreonConvexConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/AkreonConvexConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AkreonConvex)

if(PROJECT_IS_TOP_LEVEL AND (BUILD_UNIT_TESTS OR BUILD_FUNCTIONAL_TESTS))
  enable_testing()
  add_subdirectory(tests)
endif()
find_program(CLANG_FORMAT_BIN NAMES clang-format-22 clang-format)
if(PROJECT_IS_TOP_LEVEL AND CLANG_FORMAT_BIN)
  file(GLOB_RECURSE format_files CONFIGURE_DEPENDS include/*.hpp tests/*.cpp tests/*.hpp test-support/*.hpp examples/*.cpp)
  add_custom_target(clang_format_check COMMAND ${CLANG_FORMAT_BIN} --dry-run --Werror ${format_files} VERBATIM)
  add_custom_target(clang_format_fix COMMAND ${CLANG_FORMAT_BIN} -i ${format_files} VERBATIM)
endif()
