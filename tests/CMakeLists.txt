enable_testing()

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1
)

FetchContent_MakeAvailable(Catch2)

add_executable(manifolds_test manifolds_test.cpp)
add_test(NAME manifolds_test COMMAND manifolds_test)
target_link_libraries(manifolds_test PRIVATE Catch2::Catch2WithMain manifolds)
target_include_directories(manifolds_test PRIVATE ../math/include)
