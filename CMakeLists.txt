cmake_minimum_required(VERSION 3.5.0)
project(masked_conv VERSION 0.1.0 LANGUAGES CUDA CXX)

if (NOT DEFINED CXX_STANDARD)
    set(CXX_STANDARD 23)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(LIB_NAME masked_conv)

# Add src dir
include_directories("${CMAKE_SOURCE_DIR}/src")
add_library(${LIB_NAME} STATIC src/masked_conv.cu src/masked_conv.hpp)
# set_target_properties(masked_conv PROPERTIES
#                             CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(masked_conv PROPERTIES CUDA_ARCHITECTURES 86)

# Tests
enable_testing()
add_subdirectory(tests)