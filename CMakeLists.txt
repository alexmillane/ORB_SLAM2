cmake_minimum_required(VERSION 2.8.3)
project(orb_slam_2)

add_definitions(-std=c++11)

find_package(catkin_simple REQUIRED)
catkin_simple()

# Compiling DBoW2
include(${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/DBoW2/CMakeLists.txt)

# Compiling G2O
include(${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/g2o/CMakeLists.txt)

# Compiling Orb_slam_2
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeLists.txt)

cs_install()
cs_export(INCLUDE_DIRS "${DBOW2_ROOT_DIR}" "${g2o_ROOT_DIR}")
