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


###########################################################################
# Boost setup
if (LINKSTATIC)
    set (Boost_USE_STATIC_LIBS ON)
else ()
    if (MSVC)
        add_definitions (-DBOOST_ALL_DYN_LINK=1)
    endif ()
endif ()
if (BOOST_CUSTOM)
    set (Boost_FOUND true)
    # N.B. For a custom version, the caller had better set up the variables
    # Boost_VERSION, Boost_INCLUDE_DIRS, Boost_LIBRARY_DIRS, Boost_LIBRARIES.
else ()
    set (Boost_COMPONENTS filesystem system thread)
    # The FindBoost.cmake interface is broken if it uses boost's installed
    # cmake output (e.g. boost 1.70.0, cmake <= 3.14). Specifically it fails
    # to set the expected variables printed below. So until that's fixed
    # force FindBoost.cmake to use the original brute force path.
    if (NOT DEFINED Boost_NO_BOOST_CMAKE)
        set (Boost_NO_BOOST_CMAKE ON)
    endif ()
    checked_find_package (Boost REQUIRED
                       VERSION_MIN 1.55
                       COMPONENTS ${Boost_COMPONENTS}
                       RECOMMEND_MIN 1.66
                       RECOMMEND_MIN_REASON "Boost 1.66 is the oldest version our CI tests against"
                       PRINT Boost_INCLUDE_DIRS Boost_LIBRARIES
                      )
endif ()

# On Linux, Boost 1.55 and higher seems to need to link against -lrt
if (CMAKE_SYSTEM_NAME MATCHES "Linux"
      AND ${Boost_VERSION} VERSION_GREATER_EQUAL 105500)
    list (APPEND Boost_LIBRARIES "rt")
endif ()

include_directories (SYSTEM "${Boost_INCLUDE_DIRS}")
link_directories ("${Boost_LIBRARY_DIRS}")

# end Boost setup
###########################################################################


checked_find_package (ZLIB REQUIRED)  # Needed by several packages

# IlmBase & OpenEXR
checked_find_package (OpenEXR REQUIRED
                      VERSION_MIN 2.4
                      RECOMMEND_MIN 2.5
                      RECOMMEND_MIN_REASON
                        "Even extremely critical patches are no longer supplied to < 2.5"
                      PRINT IMATH_INCLUDES
                     )
# Force Imath includes to be before everything else to ensure that we have
# the right Imath/OpenEXR version, not some older version in the system
# library. This shouldn't be necessary, except for the common case of people
# building against Imath/OpenEXR 3.x when there is still a system-level
# install version of 2.x.
include_directories(BEFORE ${IMATH_INCLUDES})
if (MSVC AND NOT LINKSTATIC)
    add_definitions (-DOPENEXR_DLL) # Is this needed for new versions?
endif ()

if (OPENEXR_VERSION VERSION_GREATER_EQUAL 2.5.99)
    set (OSL_USING_IMATH 3)
else ()
    set (OSL_USING_IMATH 2)
endif ()


# OpenImageIO
checked_find_package (OpenImageIO REQUIRED
                      VERSION_MIN 2.4
                      DEFINITIONS -DOIIO_HIDE_FORMAT=1)

checked_find_package (pugixml REQUIRED
                      VERSION_MIN 1.8)


# LLVM library setup
checked_find_package (LLVM REQUIRED
                      VERSION_MIN 9.0
                      VERSION_MAX 18.9
                      PRINT LLVM_SYSTEM_LIBRARIES CLANG_LIBRARIES)
# ensure include directory is added (in case of non-standard locations
include_directories (BEFORE SYSTEM "${LLVM_INCLUDES}")
link_directories ("${LLVM_LIB_DIR}")
# Extract and concatenate major & minor, remove wayward patches,
# dots, and "svn" or other suffixes.
string (REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1\\2" OSL_LLVM_VERSION ${LLVM_VERSION})
add_definitions (-DOSL_LLVM_VERSION=${OSL_LLVM_VERSION})
add_definitions (-DOSL_LLVM_FULL_VERSION="${LLVM_VERSION}")
if (LLVM_NAMESPACE)
    add_definitions ("-DLLVM_NAMESPACE=${LLVM_NAMESPACE}")
endif ()
if (APPLE AND LLVM_VERSION VERSION_EQUAL 10.0.1 AND EXISTS "/usr/local/Cellar/llvm")
    message (WARNING
             "${ColorYellow}If you are using LLVM 10.0.1 installed by Homebrew, "
             "please note that a known bug in LLVM may produce a link error where "
             "it says it can't find libxml2.tbd. If you encounter this, please "
             "try upgrading to a newer LLVM: \n"
             "    brew upgrade llvm \n"
             "${ColorReset}\n")
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
    if (CMAKE_COMPILER_IS_CLANG AND (CLANG_VERSION_STRING VERSION_LESS 5.0
                                     OR APPLE_CLANG_VERSION_STRING VERSION_LESS 5.0))
        message (WARNING "${ColorYellow}LLVM 16+ requires clang 5.0 or higher.${ColorReset}\n")
    endif ()
endif ()

# Use opaque pointers starting with LLVM 16
if (${LLVM_VERSION} VERSION_GREATER_EQUAL 16.0)
  set(LLVM_OPAQUE_POINTERS ON)
  add_definitions (-DOSL_LLVM_OPAQUE_POINTERS)
else()
  set(LLVM_OPAQUE_POINTERS OFF)
endif()

# Enable new pass manager for LLVM 16+
if (${LLVM_VERSION} VERSION_GREATER_EQUAL 16.0)
  set(LLVM_NEW_PASS_MANAGER ON)
  add_definitions (-DOSL_LLVM_NEW_PASS_MANAGER)
else()
  set(LLVM_NEW_PASS_MANAGER OFF)
endif()


checked_find_package (partio)


# From pythonutils.cmake
find_python ()


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

        find_library(cuda_lib NAMES cudart
                    PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${CUDA_TOOLKIT_ROOT_DIR}/x64" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"
                    REQUIRED)
        set(CUDA_LIBRARIES ${cuda_lib})

        # testrender & testshade need libnvrtc
        if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "10.0")
            find_library(nvrtc_lib NAMES nvrtc
                        PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${CUDA_TOOLKIT_ROOT_DIR}/x64" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"
                        REQUIRED)
            set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${nvrtc_lib})

            set(CUDA_EXTRA_LIBS ${CUDA_EXTRA_LIBS} dl)
        endif()
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
        target_compile_definitions (${TARGET} PRIVATE "-DPTX_PATH=\"${OSL_PTX_INSTALL_DIR}\"")
        target_link_libraries (${TARGET} PRIVATE ${CUDA_LIBRARIES} ${CUDA_EXTRA_LIBS} ${OPTIX_LIBRARIES} ${OPTIX_EXTRA_LIBS})
    endfunction()
else ()
    function (osl_optix_target TARGET)
    endfunction()
endif ()
