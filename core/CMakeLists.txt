cmake_minimum_required(VERSION 3.22)

if (BUILD_CORE)
# Embedded code optimizations
add_compile_options(-ffunction-sections -fdata-sections -fvisibility=hidden -fno-rtti -flto -Oz)
# add -fno-exceptions to lib TU
endif()
