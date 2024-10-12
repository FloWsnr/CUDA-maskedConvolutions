cmake_minimum_required(VERSION 3.22)
project(masked_conv VERSION 0.1.0
        LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 23)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -v")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Add src dir
include_directories("${CMAKE_SOURCE_DIR}/src")
set(LIB_NAME mConv)

# Add core library
add_library(${LIB_NAME} STATIC src/mConv1D.cu src/mConv3D.cu src/mConv.hpp)
set_target_properties(${LIB_NAME} PROPERTIES
    CUDA_ARCHITECTURES "86"
    POSITION_INDEPENDENT_CODE ON
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-V>)
endif()

# Add profiling app
add_executable(p_mConv1D apps/profile_mConv1D.cu)
set_target_properties(p_mConv1D PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(p_mConv1D PRIVATE ${LIB_NAME})

# Tests
enable_testing()
add_subdirectory(tests)