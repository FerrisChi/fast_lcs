cmake_minimum_required(VERSION 3.20)
project(CU_FAST_LCS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_C_FLAGS_DEBUG "$ENV{CXXFLAGS} -Wall -g")
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -Wall -g")


add_executable(cu_fastlcs
    fastlcs_cu.h
    fastlcs.cu
)

add_executable(no_fastlcs
    fastlcs_nomap.h
    fastlcs_nomap.cu
)

add_executable(cpu_fastlcs
    fastlcs_cpu.h
    fastlcs.cpp
)

add_executable(profiler
    profiler.cpp
)

add_executable(dp
    dp.cpp
)

add_executable(sw
    sw.cpp
)

add_executable(gen
    gen.cpp
)