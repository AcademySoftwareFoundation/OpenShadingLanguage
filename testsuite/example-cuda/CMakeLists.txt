# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

cmake_minimum_required(VERSION 3.12)
project(examplecuda LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release")
endif ()

# Is there a better way to get the location of the main OSL CMakeLists.txt?
set(OSL_LIST_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../")
# We have to make the path absolute or CMake will fail to find the file...
get_filename_component(OSL_LIST_DIR ${OSL_LIST_DIR} ABSOLUTE)
set(CMAKE_MODULE_PATH "${OSL_LIST_DIR}/src/cmake;${OSL_LIST_DIR}/src/cmake/modules")

include(check_is_enabled)
include(checked_find_package)

find_package(OSL REQUIRED)
find_package(CUDA REQUIRED)
checked_find_package(LLVM 7.0 REQUIRED)
checked_find_package(OpenEXR 2.0 REQUIRED)
checked_find_package(OpenImageIO 2.2 REQUIRED)
checked_find_package(OptiX REQUIRED)


# Make the build area layout look like we expect
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# TODO: what are the path suffixes on other platforms?
find_library(CUDA_nvrtc_LIBRARY nvrtc HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64)
find_library(CUDA_cuda_LIBRARY cuda HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib/stubs lib64/stubs)

# TODO: move to sm_60?
set(CUDA_TARGET_ARCH sm_35)

# Compile our "renderer" to PTX
cuda_compile_ptx(CUDA_PTX_FILES cuda_grid_renderer.cu
    OPTIONS --gpu-architecture=${CUDA_TARGET_ARCH} --use_fast_math -dc --std=c++11
            --expt-relaxed-constexpr
            -I${OSL_INCLUDES}
            -I${IMATH_INCLUDES}
            -I${OpenImageIO_INCLUDES}
            -I${OPTIX_INCLUDES}
)
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/cuda_grid_renderer.ptx
    COMMAND ${CMAKE_COMMAND} -E rename ${CUDA_PTX_FILES} ${CMAKE_BINARY_DIR}/cuda_grid_renderer.ptx
    DEPENDS ${CUDA_PTX_FILES}
)
add_custom_target(cuda_grid_renderer_ptx ALL
    DEPENDS ${CMAKE_BINARY_DIR}/cuda_grid_renderer.ptx cuda_grid_renderer.cu
    SOURCES cuda_grid_renderer.cu
)

# Compile the rend_lib shadeops to PTX
cuda_compile_ptx(CUDA_PTX_FILES rend_lib.cu
    OPTIONS --gpu-architecture=${CUDA_TARGET_ARCH} --use_fast_math -dc --std=c++14
            --expt-relaxed-constexpr
            -I${OSL_INCLUDES}
            -I${IMATH_INCLUDES}
            -I${OpenImageIO_INCLUDES}
            -I${OPTIX_INCLUDES}
)
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/rend_lib.ptx
    COMMAND ${CMAKE_COMMAND} -E rename ${CUDA_PTX_FILES} ${CMAKE_BINARY_DIR}/rend_lib.ptx
    DEPENDS ${CUDA_PTX_FILES}
)
add_custom_target(rend_lib_ptx ALL
    DEPENDS ${CMAKE_BINARY_DIR}/rend_lib.ptx rend_lib.cu
    SOURCES rend_lib.cu
)

# Compile and link the main executable
add_executable(example-cuda example-cuda.cpp cuda_grid_renderer.cpp)

target_link_libraries(example-cuda PRIVATE OSL::oslexec OSL::oslquery ${CUDA_LIBRARIES} ${CUDA_nvrtc_LIBRARY} ${CUDA_cuda_LIBRARY})
target_include_directories(example-cuda PRIVATE ${CUDA_INCLUDE_DIRS})
set_property(TARGET example-cuda PROPERTY CXX_STANDARD 14)

install(TARGETS example-cuda DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
