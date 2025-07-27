include_directories(include)

add_library(math topology.cpp chart.cpp atlas.cpp smooth_map.cpp)
target_include_directories(math PUBLIC include)
