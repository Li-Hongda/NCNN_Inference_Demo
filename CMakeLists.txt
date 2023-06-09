cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

project(inference)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

message(STATUS ${CMAKE_SOURCE_DIR})

find_package(OpenCV REQUIRED)
if (OPENCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    set(ALL_LIBS ${ALL_LIBS} ${OpenCV_LIBS})
endif()

set(NCNN_DIR /home/lhd/downloads/ncnn/build/install/lib/cmake/ncnn) 
if (NOT DEFINED NCNN_DIR) 
    message(FATAL_ERROR "Can't find NCNN_DIR Var")
else()
    set(ncnn_DIR ${NCNN_DIR} CACHE PATH "Directory that contains ncnnConfig.cmake")
    find_package(ncnn REQUIRED)
    if (ncnn_FOUND) 
        set(ALL_LIBS ${ALL_LIBS} ncnn)
    endif()
endif()

include_directories(${NCNN_INCLUDE_DIRS})

aux_source_directory(${CMAKE_SOURCE_DIR}/src DIR_SRCS)
message(STATUS "Find src file: ${DIR_SRCS}")

set(EXECUTABLE_OUTPUT_PATH ../bin)
add_executable(${CMAKE_PROJECT_NAME} ${DIR_SRCS})

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(${CMAKE_PROJECT_NAME} ${ALL_LIBS})