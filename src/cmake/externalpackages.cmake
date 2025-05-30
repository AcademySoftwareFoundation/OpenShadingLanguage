# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

###########################################################################
# Find external dependencies
###########################################################################

# When not in VERBOSE mode, try to make things as quiet as possible
if (NOT VERBOSE)
    set (PkgConfig_FIND_QUIETLY true)
    set (Threads_FIND_QUIETLY true)
endif ()

message (STATUS "${ColorBoldWhite}")
message (STATUS "* Checking for dependencies...")
message (STATUS "*   - Missing a dependency 'Package'?")
message (STATUS "*     Try cmake -DPackage_ROOT=path or set environment var Package_ROOT=path")
message (STATUS "*     For many dependencies, we supply src/build-scripts/build_Package.bash")
message (STATUS "*   - To exclude an optional dependency (even if found),")
message (STATUS "*     -DUSE_Package=OFF or set environment var USE_Package=OFF ")
message (STATUS "${ColorReset}")

# Place where locally built (by src/build-scripts/*) dependencies go.
# Put it first in the prefix path.
set (OSL_LOCAL_DEPS_DIR "${CMAKE_SOURCE_DIR}/ext/dist" CACHE STRING
     "Local area for dependencies added to CMAKE_PREFIX_PATH")
list (INSERT CMAKE_PREFIX_PATH 0 ${OSL_LOCAL_DEPS_DIR})
message (STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")


include (ExternalProject)

option (BUILD_MISSING_DEPS "Try to download and build any missing dependencies" OFF)

checked_find_package (ZLIB REQUIRED)  # Needed by several packages

# IlmBase & OpenEXR
checked_find_package (Imath REQUIRED
                      VERSION_MIN 3.1
                      PRINT IMATH_INCLUDES Imath_VERSION
                     )
# Force Imath includes to be before everything else to ensure that we have
# the right Imath version, not some older version in the system library.
include_directories(BEFORE ${IMATH_INCLUDES})
set (OSL_USING_IMATH 3)


# OpenImageIO
checked_find_package (OpenImageIO REQUIRED
                      VERSION_MIN 2.5
                      DEFINITIONS OIIO_HIDE_FORMAT=1)

checked_find_package (pugixml REQUIRED
                      VERSION_MIN 1.8)


# LLVM library setup
checked_find_package (LLVM REQUIRED
                      VERSION_MIN 11.0
                      VERSION_MAX 19.9
                      PRINT LLVM_SYSTEM_LIBRARIES CLANG_LIBRARIES
                            LLVM_SHARED_MODE)
# ensure include directory is added (in case of non-standard locations
include_directories (BEFORE SYSTEM "${LLVM_INCLUDES}")
link_directories ("${LLVM_LIB_DIR}")
# Extract and concatenate major & minor, remove wayward patches,
# dots, and "svn" or other suffixes.
string (REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1\\2" OSL_LLVM_VERSION ${LLVM_VERSION})
add_compile_definitions (OSL_LLVM_VERSION=${OSL_LLVM_VERSION})
add_compile_definitions (OSL_LLVM_FULL_VERSION="${LLVM_VERSION}")
if (LLVM_NAMESPACE)
    add_compile_definitions (LLVM_NAMESPACE=${LLVM_NAMESPACE})
endif ()
if (LLVM_VERSION VERSION_GREATER_EQUAL 15.0 AND CMAKE_COMPILER_IS_CLANG
    AND ANY_CLANG_VERSION_STRING VERSION_LESS 15.0)
    message (WARNING
         "${ColorYellow}"
         "If you are using LLVM 15 or higher, you should also use clang version "
         "15 or higher, or you may get build errors.${ColorReset}\n")
endif ()
if (LLVM_VERSION VERSION_GREATER_EQUAL 16.0)
    if (CMAKE_CXX_STANDARD VERSION_LESS 17)
        message (WARNING "${ColorYellow}LLVM 16+ requires C++17 or higher. "
            "Please set CMAKE_CXX_STANDARD to 17 or higher.${ColorReset}\n")
    endif ()
    if (CMAKE_COMPILER_IS_GNUCC AND (GCC_VERSION VERSION_LESS 7.0))
        message (WARNING "${ColorYellow}LLVM 16+ requires gcc 7.0 or higher.${ColorReset}\n")
    endif ()
    if (CMAKE_COMPILER_IS_CLANG
        AND NOT (CLANG_VERSION_STRING VERSION_GREATER_EQUAL 5.0
                 OR APPLECLANG_VERSION_STRING VERSION_GREATER_EQUAL 5.0))
        message (WARNING "${ColorYellow}LLVM 16+ requires clang 5.0 or higher.${ColorReset}\n")
    endif ()
endif ()

# Use opaque pointers starting with LLVM 16
if (${LLVM_VERSION} VERSION_GREATER_EQUAL 16.0)
  set(LLVM_OPAQUE_POINTERS ON)
  add_compile_definitions (OSL_LLVM_OPAQUE_POINTERS)
else()
  set(LLVM_OPAQUE_POINTERS OFF)
endif()

# Enable new pass manager for LLVM 16+
if (${LLVM_VERSION} VERSION_GREATER_EQUAL 16.0)
  set(LLVM_NEW_PASS_MANAGER ON)
  add_compile_definitions (OSL_LLVM_NEW_PASS_MANAGER)
else()
  set(LLVM_NEW_PASS_MANAGER OFF)
endif()


checked_find_package (partio)


# From pythonutils.cmake
find_python ()
if (USE_PYTHON)
    checked_find_package (pybind11 REQUIRED VERSION_MIN 2.7)
endif ()


# Qt -- used for osltoy
option (USE_QT "Use Qt if found" ON)
if (USE_QT)
    checked_find_package (Qt6 COMPONENTS Core Gui Widgets OpenGLWidgets)
    if (NOT Qt6_FOUND)
        checked_find_package (Qt5 COMPONENTS Core Gui Widgets OpenGL)
    endif ()
    if (NOT Qt5_FOUND AND NOT Qt6_FOUND AND APPLE)
        message (STATUS "  If you think you installed qt with Homebrew and it still doesn't work,")
        message (STATUS "  try:   export PATH=/usr/local/opt/qt/bin:$PATH")
    endif ()
endif ()


# CUDA setup
option (CUDA_PREFER_STATIC_LIBS "Prefer static CUDA libraries" OFF)
if (OSL_USE_OPTIX)
    if (USE_LLVM_BITCODE)
        if (NOT CUDA_TOOLKIT_ROOT_DIR AND NOT $ENV{CUDA_TOOLKIT_ROOT_DIR} STREQUAL "")
            set (CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_TOOLKIT_ROOT_DIR})
        endif ()

        if (NOT CUDA_FIND_QUIETLY OR NOT OptiX_FIND_QUIETLY)
            message (STATUS "CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
        endif ()

        checked_find_package (CUDA REQUIRED
                             VERSION_MIN 9.0
                             RECOMMEND_MIN 11.0
                             RECOMMEND_MIN_REASON
                                "We don't actively test CUDA older than 11"
                             PRINT CUDA_INCLUDES)
        set (CUDA_INCLUDES ${CUDA_TOOLKIT_ROOT_DIR}/include)
        include_directories (BEFORE "${CUDA_INCLUDES}")

        STRING (FIND ${LLVM_TARGETS} "NVPTX" nvptx_index)
        if (NOT ${nvptx_index} GREATER -1)
            message (FATAL_ERROR "NVPTX target is not available in the provided LLVM build")
        endif()

        set (CUDA_LIB_FLAGS "--cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")

        # If the user wants, try to use static libs here to putting static lib
        # suffixes earlier in the suffix list. Don't forget to restore after
        # so that this only applies to these library searches right here.
        set (save_lib_path ${CMAKE_FIND_LIBRARY_SUFFIXES})
        if (CUDA_PREFER_STATIC_LIBS)
            set (CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
            find_library(cudart_lib REQUIRED
                         NAMES cudart_static cudart
                         PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${CUDA_TOOLKIT_ROOT_DIR}/x64" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
        else ()
            find_library(cudart_lib REQUIRED
                         NAMES cudart
                         PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${CUDA_TOOLKIT_ROOT_DIR}/x64" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
        endif ()
        # Is it really a good idea to completely reset CUDA_LIBRARIES here?
        set(CUDA_LIBRARIES ${cudart_lib})
        set(CUDA_EXTRA_LIBS ${CUDA_EXTRA_LIBS} dl rt)
        set (CMAKE_FIND_LIBRARY_SUFFIXES ${save_lib_path})
        unset (save_lib_path)
    endif()

    # OptiX setup
    if (OSL_USE_OPTIX AND OSL_BUILD_TESTS)
        checked_find_package (OptiX REQUIRED
                              VERSION_MIN 7.0)
        include_directories (BEFORE "${OPTIX_INCLUDES}")
        if (NOT USE_LLVM_BITCODE OR NOT USE_FAST_MATH)
            message (FATAL_ERROR "Enabling OptiX requires USE_LLVM_BITCODE=1 and USE_FAST_MATH=1")
        endif ()
    endif ()

    function (osl_optix_target TARGET)
        target_include_directories (${TARGET} BEFORE PRIVATE ${CUDA_INCLUDES} ${OPTIX_INCLUDES})
        ## XXX: Should -DPTX_PATH point to (or include) CMAKE_CURRENT_BINARY_DIR so tests can run before installation ?
        target_compile_definitions (${TARGET} PRIVATE PTX_PATH="${OSL_PTX_INSTALL_DIR}")
        target_link_libraries (${TARGET} PRIVATE ${CUDA_LIBRARIES} ${CUDA_EXTRA_LIBS} ${OPTIX_LIBRARIES} ${OPTIX_EXTRA_LIBS})
    endfunction()
else ()
    message(STATUS "CUDA/OptiX support disabled")
    function (osl_optix_target TARGET)
    endfunction()
endif ()


###########################################################################
# Tessil/robin-map

option (BUILD_ROBINMAP_FORCE "Force local download/build of robin-map even if installed" OFF)
option (BUILD_MISSING_ROBINMAP "Local download/build of robin-map if not installed" ON)
set (BUILD_ROBINMAP_VERSION "v0.6.2" CACHE STRING "Preferred Tessil/robin-map version, of downloading/building our own")

macro (find_or_download_robin_map)
    # If we weren't told to force our own download/build of robin-map, look
    # for an installed version. Still prefer a copy that seems to be
    # locally installed in this tree.
    if (NOT BUILD_ROBINMAP_FORCE)
        find_package (Robinmap QUIET)
    endif ()
    # If an external copy wasn't found and we requested that missing
    # packages be built, or we we are forcing a local copy to be built, then
    # download and build it.
    # Download the headers from github
    if ((BUILD_MISSING_ROBINMAP AND NOT ROBINMAP_FOUND) OR BUILD_ROBINMAP_FORCE)
        message (STATUS "Downloading local Tessil/robin-map")
        set (ROBINMAP_INSTALL_DIR "${PROJECT_SOURCE_DIR}/ext/robin-map")
        set (ROBINMAP_GIT_REPOSITORY "https://github.com/Tessil/robin-map")
        if (NOT IS_DIRECTORY ${ROBINMAP_INSTALL_DIR}/include/tsl)
            find_package (Git REQUIRED)
            execute_process(COMMAND             ${GIT_EXECUTABLE} clone    ${ROBINMAP_GIT_REPOSITORY} -n ${ROBINMAP_INSTALL_DIR})
            execute_process(COMMAND             ${GIT_EXECUTABLE} checkout ${BUILD_ROBINMAP_VERSION}
                            WORKING_DIRECTORY   ${ROBINMAP_INSTALL_DIR})
            if (IS_DIRECTORY ${ROBINMAP_INSTALL_DIR}/include/tsl)
                message (STATUS "DOWNLOADED Tessil/robin-map to ${ROBINMAP_INSTALL_DIR}.\n"
                         "Remove that dir to get rid of it.")
            else ()
                message (FATAL_ERROR "Could not download Tessil/robin-map")
            endif ()
        endif ()
        set (ROBINMAP_INCLUDE_DIR "${ROBINMAP_INSTALL_DIR}/include")
    endif ()
    checked_find_package (Robinmap REQUIRED)
endmacro()

find_or_download_robin_map ()
