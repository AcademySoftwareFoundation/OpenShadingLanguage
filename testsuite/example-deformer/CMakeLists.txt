# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

cmake_minimum_required (VERSION 3.12)
project (osldeformer
         LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release")
endif ()

message (STATUS "Building ${PROJECT_NAME} ${PROJECT_VERSION} - ${CMAKE_BUILD_TYPE}")

# Make the build area layout look like we expect
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Use C++11
set (CMAKE_CXX_STANDARD 11 CACHE STRING "C++ standard to prefer (11, 14, 17, etc.)")
set (CMAKE_CXX_STANDARD_REQUIRD ON)
set (CMAKE_CXX_EXTENSIONS OFF)


# Make sure we have dependencies we need
find_package (OSL REQUIRED)


add_executable(osldeformer osldeformer.cpp)
target_link_libraries (osldeformer
                       PRIVATE OSL::oslexec OSL::oslquery)
