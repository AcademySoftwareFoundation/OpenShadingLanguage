# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Keep this in sync with the top-level project.
cmake_minimum_required (VERSION 3.23)
project (examplecuda LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release")
endif ()

# Is there a better way to get the location of the main OSL CMakeLists.txt?
set (OSL_LIST_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../")
# We have to make the path absolute or CMake will fail to find the file...
get_filename_component (OSL_LIST_DIR "${OSL_LIST_DIR}" ABSOLUTE)
list (APPEND CMAKE_MODULE_PATH
      "${OSL_LIST_DIR}/src/cmake"
      "${OSL_LIST_DIR}/src/cmake/modules")

find_package (OSL REQUIRED)
find_package (CUDAToolkit 9.0 REQUIRED)
find_package (OptiX 7.0 REQUIRED)


# Make the build area layout look like we expect
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
    set (cuda_default_arch "sm_75")
else ()
    set (cuda_default_arch "sm_60")
endif ()
set (CUDA_TARGET_ARCH "${cuda_default_arch}" CACHE STRING
     "CUDA GPU architecture (e.g. sm_60)")

set (CMAKE_CXX_STANDARD 17 CACHE STRING
     "C++ standard to build with (17, 20, etc.)")

get_target_property (imath_include_dirs Imath::Imath
                     INTERFACE_INCLUDE_DIRECTORIES)
get_target_property (oiio_include_dirs OpenImageIO::OpenImageIO
                     INTERFACE_INCLUDE_DIRECTORIES)
set (cuda_toolkit_include_dirs ${CUDAToolkit_INCLUDE_DIRS})
foreach (cuda_include_dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
    if (IS_DIRECTORY "${cuda_include_dir}/cccl")
        list (APPEND cuda_toolkit_include_dirs "${cuda_include_dir}/cccl")
    endif ()
endforeach ()
list (FILTER cuda_toolkit_include_dirs EXCLUDE REGEX "^$")
list (REMOVE_DUPLICATES cuda_toolkit_include_dirs)
set (cuda_ptx_include_dirs ${OSL_INCLUDES}
                           ${imath_include_dirs}
                           ${oiio_include_dirs}
                           ${OPTIX_INCLUDES}
                           ${cuda_toolkit_include_dirs})
list (TRANSFORM cuda_ptx_include_dirs PREPEND -I)

# Compile a CUDA source file to PTX with NVCC. This replaces the helper that
# was supplied by the deprecated FindCUDA module.
function (compile_cuda_to_ptx cuda_src ptx_generated)
    get_filename_component (cuda_src_we "${cuda_src}" NAME_WE)
    set (cuda_ptx "${CMAKE_CURRENT_BINARY_DIR}/${cuda_src_we}.ptx")
    file (GLOB cuda_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

    add_custom_command (OUTPUT "${cuda_ptx}"
        COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
                --gpu-architecture=${CUDA_TARGET_ARCH}
                --use_fast_math
                --std=c++${CMAKE_CXX_STANDARD}
                --expt-relaxed-constexpr
                -dc -ptx
                ${cuda_ptx_include_dirs}
                "${CMAKE_CURRENT_SOURCE_DIR}/${cuda_src}"
                -o "${cuda_ptx}"
        MAIN_DEPENDENCY "${cuda_src}"
        DEPENDS ${cuda_headers}
        VERBATIM)

    set (${ptx_generated} "${cuda_ptx}" PARENT_SCOPE)
endfunction ()

compile_cuda_to_ptx (cuda_grid_renderer.cu cuda_grid_renderer_ptx_file)
add_custom_target (cuda_grid_renderer_ptx ALL
                   DEPENDS "${cuda_grid_renderer_ptx_file}"
                   SOURCES cuda_grid_renderer.cu)

compile_cuda_to_ptx (rend_lib.cu rend_lib_ptx_file)
add_custom_target (rend_lib_ptx ALL
                   DEPENDS "${rend_lib_ptx_file}"
                   SOURCES rend_lib.cu)

# Compile and link the main executable
add_executable (example-cuda example-cuda.cpp cuda_grid_renderer.cpp)
add_dependencies (example-cuda cuda_grid_renderer_ptx rend_lib_ptx)

target_link_libraries (example-cuda PRIVATE
                       OSL::oslexec
                       OSL::oslquery
                       CUDA::cudart
                       CUDA::nvrtc
                       CUDA::cuda_driver)
target_include_directories (example-cuda PRIVATE ${cuda_toolkit_include_dirs})
set_property (TARGET example-cuda PROPERTY CXX_STANDARD 17)

install (TARGETS example-cuda DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
