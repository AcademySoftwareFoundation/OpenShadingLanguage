# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

cmake_minimum_required (VERSION 3.23)
project (tinyrender
         LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release")
endif ()

message (STATUS "Building ${PROJECT_NAME}...")

# Make the build area layout look like we expect
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set (CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to prefer (17, 20, etc.)")
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)


# Make sure we have dependencies we need
find_package (OSL REQUIRED)


# Compile the shaders with the oslc that was installed along with OSL.
set (shader_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set (shader_oso_dir "${CMAKE_BINARY_DIR}/shaders")
file (MAKE_DIRECTORY "${shader_oso_dir}")

set (compiled_shaders)
foreach (shader matte metal glass plastic)
    set (oso "${shader_oso_dir}/${shader}.oso")
    add_custom_command (
        OUTPUT "${oso}"
        COMMAND "$<TARGET_FILE:OSL::oslc>"
            -I "${shader_src_dir}" -o "${oso}" "${shader_src_dir}/${shader}.osl"
        MAIN_DEPENDENCY "${shader_src_dir}/${shader}.osl"
        DEPENDS "${shader_src_dir}/rtiow.h" OSL::oslc
        COMMENT "Compiling OSL shader ${shader}.osl"
        VERBATIM)
    list (APPEND compiled_shaders "${oso}")
endforeach ()
add_custom_target (tinyrender_shaders ALL DEPENDS ${compiled_shaders})


add_executable (tinyrender
                tinyrender.cpp
                osl/shading.cpp)
add_dependencies (tinyrender tinyrender_shaders)
target_include_directories (tinyrender
                            PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions (tinyrender
                            PRIVATE TINYRENDER_SHADER_DIR="${shader_oso_dir}")
# OSL::oslexec brings in OpenImageIO and Imath.
target_link_libraries (tinyrender
                       PRIVATE OSL::oslexec)
