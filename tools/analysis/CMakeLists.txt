cmake_minimum_required(VERSION 3.1)
project(bonsai_analysis CXX)
find_package(MPI)
find_package(OpenMP)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(OpenMP_CXX_FLAGS "-fopenmp -I/usr/lib64/gcc/x86_64-suse-linux/7/include") 
set(OpenMP_CXX_LIB_NAMES "omp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_executable(bar_buckling_ana bar_buckling.cpp)
target_link_libraries(bar_buckling_ana PRIVATE ${MPI_LIBRARIES})
target_include_directories(bar_buckling_ana PRIVATE ${MPI_INCLUDE_PATH})
