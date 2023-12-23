cmake_minimum_required(VERSION 2.8)
project(satelliteIO CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(MPI)

add_executable(comb_amuse_bonsai comb_amuse_bonsai.cpp)
target_link_libraries(comb_amuse_bonsai PRIVATE ${MPI_LIBRARIES})
target_include_directories(comb_amuse_bonsai PRIVATE ${MPI_INCLUDE_PATH})

add_executable(cvt_bonsai2dumbp cvt_bonsai2dumbp.cpp)
target_link_libraries(cvt_bonsai2dumbp PRIVATE ${MPI_LIBRARIES})
target_include_directories(cvt_bonsai2dumbp PRIVATE ${MPI_INCLUDE_PATH})

add_executable(readBonsai readBonsai.cpp)
target_link_libraries(readBonsai PRIVATE ${MPI_LIBRARIES})
target_include_directories(readBonsai PRIVATE ${MPI_INCLUDE_PATH})

add_executable(cvt_tipsy2bonsai cvt_tipsy_gtc12_2bonsai.cpp)
target_link_libraries(cvt_tipsy2bonsai PRIVATE ${MPI_LIBRARIES})
target_include_directories(cvt_tipsy2bonsai PRIVATE ${MPI_INCLUDE_PATH})

add_executable(cvt_tipsy2ascii cvt_tipsy2ascii.cpp)
target_link_libraries(cvt_tipsy2ascii PRIVATE ${MPI_LIBRARIES})
target_include_directories(cvt_tipsy2ascii PRIVATE ${MPI_INCLUDE_PATH})

add_executable(cvt_ascii2tipsy cvt_ascii2tipsy.cpp tipsyIO.cpp)
target_link_libraries(cvt_ascii2tipsy PRIVATE ${MPI_LIBRARIES})
target_include_directories(cvt_ascii2tipsy PRIVATE ${MPI_INCLUDE_PATH})

add_executable(cvt_bonsai2tipsy cvt_bonsai2tipsy.cpp tipsyIO.cpp)
target_link_libraries(cvt_bonsai2tipsy PRIVATE ${MPI_LIBRARIES})
target_include_directories(cvt_bonsai2tipsy PRIVATE ${MPI_INCLUDE_PATH})
