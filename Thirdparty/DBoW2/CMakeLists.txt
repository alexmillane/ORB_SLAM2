#cmake_minimum_required(VERSION 2.8)
#project(DBoW2)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

set(DBOW2_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

set(HDRS_DBOW2
  ${DBOW2_ROOT_DIR}/DBoW2/BowVector.h
  ${DBOW2_ROOT_DIR}/DBoW2/FORB.h 
  ${DBOW2_ROOT_DIR}/DBoW2/FClass.h       
  ${DBOW2_ROOT_DIR}/DBoW2/FeatureVector.h
  ${DBOW2_ROOT_DIR}/DBoW2/ScoringObject.h   
  ${DBOW2_ROOT_DIR}/DBoW2/TemplatedVocabulary.h)
set(SRCS_DBOW2
  ${DBOW2_ROOT_DIR}/DBoW2/BowVector.cpp
  ${DBOW2_ROOT_DIR}/DBoW2/FORB.cpp      
  ${DBOW2_ROOT_DIR}/DBoW2/FeatureVector.cpp
  ${DBOW2_ROOT_DIR}/DBoW2/ScoringObject.cpp)

set(HDRS_DUTILS
  ${DBOW2_ROOT_DIR}/DUtils/Random.h
  ${DBOW2_ROOT_DIR}/DUtils/Timestamp.h)
set(SRCS_DUTILS
  ${DBOW2_ROOT_DIR}/DUtils/Random.cpp
  ${DBOW2_ROOT_DIR}/DUtils/Timestamp.cpp)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

include_directories(${OpenCV_INCLUDE_DIRS} ${DBOW2_ROOT_DIR}/DBoW2)
add_library(DBoW2 ${SRCS_DBOW2} ${SRCS_DUTILS})
target_link_libraries(DBoW2 ${OpenCV_LIBS})

#cs_install()
