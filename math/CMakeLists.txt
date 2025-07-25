include_directories(include)

add_library(manifolds topology.cpp chart.cpp atlas.cpp)
target_include_directories(manifolds PUBLIC include)
