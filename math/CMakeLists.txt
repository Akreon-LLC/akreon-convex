include_directories(include)

add_library(manifolds atlas.cpp chart.cpp topology.cpp)
target_include_directories(manifolds PUBLIC include)
