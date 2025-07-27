enable_testing()

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1
)

FetchContent_MakeAvailable(Catch2)

add_executable(math_test math_test.cpp)
add_test(NAME math_test COMMAND math_test)
target_link_libraries(math_test PRIVATE Catch2::Catch2WithMain math)
target_include_directories(math_test PRIVATE ../math/include)
