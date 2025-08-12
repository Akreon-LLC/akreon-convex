include_directories(include)

add_library(math math_iface.cpp)
target_link_libraries(math)
target_include_directories(math PUBLIC include)
