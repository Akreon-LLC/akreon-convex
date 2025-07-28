FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        master
)

FetchContent_MakeAvailable(Eigen3)

include_directories(include)

add_library(math SHARED math_iface.cpp)
target_link_libraries(math PRIVATE Eigen3::Eigen)
target_include_directories(math PUBLIC include)
