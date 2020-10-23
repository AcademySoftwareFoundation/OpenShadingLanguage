# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

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
    if (NOT USE_STD_REGEX)
        list (APPEND Boost_COMPONENTS regex)
    endif ()
    # The FindBoost.cmake interface is broken if it uses boost's installed
    # cmake output (e.g. boost 1.70.0, cmake <= 3.14). Specifically it fails
    # to set the expected variables printed below. So until that's fixed
    # force FindBoost.cmake to use the original brute force path.
    set (Boost_NO_BOOST_CMAKE ON)
    checked_find_package (Boost 1.55 REQUIRED
                       COMPONENTS ${Boost_COMPONENTS}
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
checked_find_package (OpenEXR 2.0 REQUIRED)
# We use Imath so commonly, may as well include it everywhere.
include_directories ("${OPENEXR_INCLUDES}" "${ILMBASE_INCLUDES}"
                     "${ILMBASE_INCLUDES}/OpenEXR")
if (CMAKE_COMPILER_IS_CLANG AND OPENEXR_VERSION VERSION_LESS 2.3)
    # clang C++ >= 11 doesn't like 'register' keyword in old exr headers
    add_compile_options (-Wno-deprecated-register)
endif ()
if (MSVC AND NOT LINKSTATIC)
    add_definitions (-DOPENEXR_DLL) # Is this needed for new versions?
endif ()


# OpenImageIO
checked_find_package (OpenImageIO 2.1.9 REQUIRED)


checked_find_package (pugixml 1.8 REQUIRED)


# LLVM library setup

############ HACK ##############
# On OSX, the Homebrew (and maybe any build) of LLVM 10.0 seems to have a
# link conflict with its dependency on the llvm libc++ and the system
# libc++, both can end up dynamically linked and lead to very subtle and
# frustrating behavior failures (in particular, osl's use of libclang will
# botch include file parsing any time LD_LIBRARY_PATH doesn't have the llvm
# libc++ first).
#
# It seems that this is not a problem when linking against the llvm and
# libclang libraries statically. So on apple and when LLVM 10+ are involved,
# just force that choice. Other than larger executables, it seems harmless,
# and in any case a better choice than this beastly bug.
#
# We can periodically revisit this with new version of LLVM, maybe they will
# fix things and we won't require this preemptive static linking.
if (APPLE AND LLVM_VERSION VERSION_GREATER_EQUAL 10.0)
    set (LLVM_STATIC ON)
endif ()

checked_find_package(LLVM REQUIRED CONFIG
                     HINTS ${LLVM_DIRECTORY})

# manual version check because "LLVM is API-compatible only with matching major.minor versions" aka it requires VERSION_EQUAL
if ( ${LLVM_PACKAGE_VERSION} VERSION_LESS 7 )
	message(FATAL_ERROR "LLVM 7.0+ is required but ${LLVM_PACKAGE_VERSION} was found")
endif()

include_directories(BEFORE SYSTEM ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
link_directories(${LLVM_LIBRARY_DIRS})

llvm_map_components_to_libnames(LLVM_LIBRARIES 
	MCJIT Passes 
	${LLVM_TARGETS_TO_BUILD})   #this will link with all LLVM targets (x86, nvptx, ...) specific libraries (codegen, dissasembly,...)

# Extract and concatenate major & minor, remove wayward patches,
# dots, and "svn" or other suffixes.
string (REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1\\2" OSL_LLVM_VERSION ${LLVM_PACKAGE_VERSION})
add_definitions (-DOSL_LLVM_VERSION=${OSL_LLVM_VERSION})
add_definitions (-DOSL_LLVM_FULL_VERSION="${LLVM_PACKAGE_VERSION}")
if (LLVM_NAMESPACE)
	add_definitions ("-DLLVM_NAMESPACE=${LLVM_NAMESPACE}")
endif ()
if (NOT LLVM_DIRECTORY)
	set(LLVM_DIRECTORY ${LLVM_INSTALL_PREFIX})
endif()
set(LLVM_TARGETS ${LLVM_TARGETS_TO_BUILD})

checked_find_package(Clang REQUIRED)
include_directories(BEFORE SYSTEM ${CLANG_INCLUDE_DIRS}) 
set(CLANG_LIBRARIES clangCrossTU) # TODO figure out what exactly we need (this seems to work though it might be overkill)	

if (LLVM_VERSION VERSION_GREATER_EQUAL 10.0.0 AND
    CMAKE_CXX_STANDARD VERSION_LESS 14)
    message (FATAL_ERROR
             "LLVM 10+ requires C++14 or higher (was ${CMAKE_CXX_STANDARD}). "
             "To build against this LLVM ${LLVM_VERSION}, you need to set "
             "build option CMAKE_CXX_STANDARD=14. The minimum requirements "
             "for that are gcc >= 5.1, clang >= 3.5, Apple clang >= 7, "
             "icc >= 7, MSVS >= 2017. "
             "If you must use C++11, you need to build against LLVM 9 or earlier.")
endif ()
if (APPLE AND LLVM_VERSION VERSION_EQUAL 10.0.1 AND EXISTS "/usr/local/Cellar/llvm")
    message (WARNING
             "${ColorYellow}If you are using LLVM 10.0.1 installed by Homebrew, "
             "please note that a known bug in LLVM may produce a link error where "
             "it says it can't find libxml2.tbd. If you encounter this, please "
             "try downgrading to LLVM 9: \n"
             "    brew uninstall llvm \n"
             "    brew install llvm@9 \n"
             "    export LLVM_DIRECTORY=/usr/local/opt/llvm@9 "
             "${ColorReset}\n")
endif ()


# partio
checked_find_package (partio)


# From pythonutils.cmake
find_python ()


# Qt -- used for osltoy
set (qt5_modules Core Gui Widgets)
if (OPENGL_FOUND)
    list (APPEND qt5_modules OpenGL)
endif ()
option (USE_QT "Use Qt if found" ON)
checked_find_package (Qt5 COMPONENTS ${qt5_modules})
if (USE_QT AND NOT Qt5_FOUND AND APPLE)
    message (STATUS "  If you think you installed qt5 with Homebrew and it still doesn't work,")
    message (STATUS "  try:   export PATH=/usr/local/opt/qt5/bin:$PATH")
endif ()


# CUDA setup
if (USE_CUDA OR USE_OPTIX)
    if (NOT CUDA_TOOLKIT_ROOT_DIR AND NOT $ENV{CUDA_TOOLKIT_ROOT_DIR} STREQUAL "")
        set (CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_TOOLKIT_ROOT_DIR})
    endif ()

    if (NOT CUDA_FIND_QUIETLY OR NOT OptiX_FIND_QUIETLY)
        message (STATUS "CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
    endif ()

    checked_find_package (CUDA 8.0 REQUIRED
                          PRINT CUDA_INCLUDES)
    set (CUDA_INCLUDES ${CUDA_TOOLKIT_ROOT_DIR}/include)
    include_directories (BEFORE "${CUDA_INCLUDES}")

    STRING (FIND ${LLVM_TARGETS} "NVPTX" nvptx_index)
    if (NOT ${nvptx_index} GREATER -1)
        message (FATAL_ERROR "NVPTX target is not available in the provided LLVM build")
    endif()

    if (${CUDA_VERSION} VERSION_GREATER 8 AND ${LLVM_VERSION} VERSION_LESS 6)
        message (FATAL_ERROR "CUDA ${CUDA_VERSION} requires LLVM 6.0 or greater")
    endif ()

    set (CUDA_LIB_FLAGS "--cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")

    find_library(cuda_lib NAMES cudart
                 PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${CUDA_TOOLKIT_ROOT_DIR}/x64"
                 REQUIRED)
    set(CUDA_LIBRARIES ${cuda_lib})

    # testrender & testshade need libnvrtc
    if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "10.0")
        find_library(nvrtc_lib NAMES nvrtc
                     PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${CUDA_TOOLKIT_ROOT_DIR}/x64"
                     REQUIRED)
        set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${nvrtc_lib})

        set(CUDA_EXTRA_LIBS ${CUDA_EXTRA_LIBS} dl)
    endif()

    # OptiX setup
    if (USE_OPTIX)
        checked_find_package (OptiX 5.1 REQUIRED)
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
