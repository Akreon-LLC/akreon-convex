include_directories(include)

add_library(manifolds topology.cpp chart.cpp atlas.cpp smooth_map.cpp)
target_include_directories(manifolds PUBLIC include)
