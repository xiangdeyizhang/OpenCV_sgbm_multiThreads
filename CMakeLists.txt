cmake_minimum_required(VERSION 2.8)

project(tracker)
  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
if(CMAKE_COMPILER_IS_GNUCXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wextra -pedantic -Wno-long-long")
SET(CMAKE_C_FLAGS_RELEASE "$ENV{CFLAGS} -O2 -Wall -std=c99")
endif()

#FIND_PACKAGE( OpenCV REQUIRED )
#link_directories ( ${OpenCV_LIB_DIR} )
# 寻找OpenCV库
set(OpenCV_DIR /home/xz/opencv4.3/build)
find_package( OpenCV 4 REQUIRED )
# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )
MESSAGE(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR}")
MESSAGE(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

include_directories(/home/xz/ncnn/build/install/include)
link_directories(/home/xz/ncnn/build/install/lib)


if (OPENMP_FOUND)
    message("OpenMP found!")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#include_directories ( ${OpenCV_INCLUDE_DIRS} )


add_executable(MSGBM MultiThreshadsSGBM.cpp )
target_link_libraries(MSGBM ncnn ${OpenCV_LIBS}  -fopenmp )





