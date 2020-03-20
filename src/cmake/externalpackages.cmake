# Copyright 2009-present Sony Pictures Imageworks, et al.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage/blob/master/LICENSE

###########################################################################
# Find external dependencies
###########################################################################

# When not in VERBOSE mode, try to make things as quiet as possible
if (NOT VERBOSE)
    set (Bison_FIND_QUIETLY true)
    set (Boost_FIND_QUIETLY true)
    set (Curses_FIND_QUIETLY true)
    set (Flex_FIND_QUIETLY true)
    # set (LLVM_FIND_QUIETLY true)
    set (OpenEXR_FIND_QUIETLY true)
    # set (OpenImageIO_FIND_QUIETLY true)
    # set (Partio_FIND_QUIETLY true)
    set (PkgConfig_FIND_QUIETLY true)
    set (PugiXML_FIND_QUIETLY TRUE)
    set (PythonInterp_FIND_QUIETLY true)
    set (PythonLibs_FIND_QUIETLY true)
    set (Qt5_FIND_QUIETLY true)
    set (Threads_FIND_QUIETLY true)
    set (ZLIB_FIND_QUIETLY true)
    set (CUDA_FIND_QUIETLY true)
    set (OptiX_FIND_QUIETLY true)
endif ()

message (STATUS "${ColorBoldWhite}")
message (STATUS "* Checking for dependencies...")
message (STATUS "*   - Missing a dependency 'Package'?")
message (STATUS "*     Try cmake -DPackage_ROOT=path or set environment var Package_ROOT=path")
message (STATUS "*     For many dependencies, we supply src/build-scripts/build_Package.bash")
message (STATUS "*   - To exclude an optional dependency (even if found),")
message (STATUS "*     -DUSE_Package=OFF or set environment var USE_Package=OFF ")
message (STATUS "${ColorReset}")

set (OSL_LOCAL_DEPS_PATH "${CMAKE_SOURCE_DIR}/ext/dist" CACHE STRING
     "Local area for dependencies added to CMAKE_PREFIX_PATH")
list (APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/ext/dist)

set (REQUIED_DEPS "" CACHE STRING
     "Additional dependencies to consider required (semicolon-separated list, or ALL)")
set (OPTIONAL_DEPS "" CACHE STRING
     "Additional dependencies to consider optional (semicolon-separated list, or ALL)")


# checked_find_package(pkgname ..) is a wrapper for find_package, with the
# following extra features:
#   * If either `USE_pkgname` or the all-uppercase `USE_PKGNAME` (or
#     `ENABLE_pkgname` or `ENABLE_PKGNAME`) exists as either a CMake or
#     environment variable, is nonempty by contains a non-true/nonnzero
#     value, do not search for or use the package. The optional ENABLE <var>
#     arguments allow you to override the name of the enabling variable. In
#     other words, support for the dependency is presumed to be ON, unless
#     turned off explicitly from one of these sources.
#   * Print a message if the package is enabled but not found. This is based
#     on ${pkgname}_FOUND or $PKGNNAME_FOUND.
#   * Optional DEFINITIONS <string> are passed to add_definitions if the
#     package is found.
#   * Optional PRINT <list> is a list of variables that will be printed
#     if the package is found, if VERBOSE is on.
#   * Optional DEPS <list> is a list of hard dependencies; for each one, if
#     dep_FOUND is not true, disable this package with an error message.
#   * Optional ISDEPOF <downstream> names another package for which the
#     present package is only needed because it's a dependency, and
#     therefore if <downstream> is disabled, we don't bother with this
#     package either.
#
# N.B. This needs to be a macro, not a function, because the find modules
# will set(blah val PARENT_SCOPE) and we need that to be the global scope,
# not merely the scope for this function.
macro (checked_find_package pkgname)
    cmake_parse_arguments(_pkg "REQUIRED" "ENABLE;ISDEPOF" "DEFINITIONS;PRINT;DEPS" ${ARGN})
        # Arguments: <prefix> noValueKeywords singleValueKeywords multiValueKeywords argsToParse
    string (TOUPPER ${pkgname} pkgname_upper)
    if (NOT VERBOSE)
        set (${pkgname}_FIND_QUIETLY true)
        set (${pkgname_upper}_FIND_QUIETLY true)
    endif ()
    if ("${pkgname}" IN_LIST REQUIRED_DEPS OR "ALL" IN_LIST REQUIRED_DEPS)
        set (_pkg_REQUIRED 1)
    endif ()
    if ("${pkgname}" IN_LIST OPTIONAL_DEPS OR "ALL" IN_LIST OPTIONAL_DEPS)
        set (_pkg_REQUIRED 0)
    endif ()
    set (_quietskip false)
    check_is_enabled (${pkgname} _enable)
    set (_disablereason "")
    foreach (_dep ${_pkg_DEPS})
        if (_enable AND NOT ${_dep}_FOUND)
            set (_enable false)
            set (ENABLE_${pkgname} OFF PARENT_SCOPE)
            set (_disablereason "(because ${_dep} was not found)")
        endif ()
    endforeach ()
    if (_pkg_ISDEPOF)
        check_is_enabled (${_pkg_ISDEPOF} _dep_enabled)
        if (NOT _dep_enabled)
            set (_enable false)
            set (_quietskip true)
        endif ()
    endif ()
    if (_enable OR _pkg_REQUIRED)
        find_package (${pkgname} ${_pkg_UNPARSED_ARGUMENTS})
        if (${pkgname}_FOUND OR ${pkgname_upper}_FOUND)
            foreach (_vervar ${pkgname_upper}_VERSION ${pkgname}_VERSION_STRING
                             ${pkgname_upper}_VERSION_STRING)
                if (NOT ${pkgname}_VERSION AND ${_vervar})
                    set (${pkgname}_VERSION ${${_vervar}})
                endif ()
            endforeach ()
            message (STATUS "${ColorGreen}Found ${pkgname} ${${pkgname}_VERSION} ${ColorReset}")
            if (VERBOSE)
                set (_vars_to_print ${pkgname}_INCLUDES ${pkgname_upper}_INCLUDES
                                    ${pkgname}_INCLUDE_DIR ${pkgname_upper}_INCLUDE_DIR
                                    ${pkgname}_INCLUDE_DIRS ${pkgname_upper}_INCLUDE_DIRS
                                    ${pkgname}_LIBRARIES ${pkgname_upper}_LIBRARIES
                                    ${_pkg_PRINT})
                list (REMOVE_DUPLICATES _vars_to_print)
                foreach (_v IN LISTS _vars_to_print)
                    if (NOT "${${_v}}" STREQUAL "")
                        message (STATUS "    ${_v} = ${${_v}}")
                    endif ()
                endforeach ()
            endif ()
            add_definitions (${_pkg_DEFINITIONS})
        else ()
            message (STATUS "${ColorRed}${pkgname} library not found ${ColorReset}")
            if (${pkgname}_ROOT)
                message (STATUS "${ColorRed}    ${pkgname}_ROOT was: ${${pkgname}_ROOT} ${ColorReset}")
            elseif ($ENV{${pkgname}_ROOT})
                message (STATUS "${ColorRed}    ENV ${pkgname}_ROOT was: ${${pkgname}_ROOT} ${ColorReset}")
            else ()
                message (STATUS "${ColorRed}    Try setting ${pkgname}_ROOT ? ${ColorReset}")
            endif ()
            if (EXISTS "${PROJECT_SOURCE_DIR}/src/build-scripts/build_${pkgname}.bash")
                message (STATUS "${ColorRed}    Maybe this will help:  src/build-scripts/build_${pkgname}.bash ${ColorReset}")
            endif ()
            if (_pkg_REQUIRED)
                message (FATAL_ERROR "${ColorRed}${pkgname} is required, aborting.${ColorReset}")
            endif ()
        endif()
    else ()
        if (NOT _quietskip)
            message (STATUS "${ColorRed}Not using ${pkgname} -- disabled ${_disablereason} ${ColorReset}")
        endif ()
    endif ()
endmacro()




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
set (OIIO_LIBNAME_SUFFIX "" CACHE STRING
     "Optional name appended to OIIO libraries that are built")
checked_find_package (OpenImageIO 2.0 REQUIRED
                      PRINT OIIOTOOL_BIN)
if (OPENIMAGEIO_FOUND)
    include_directories ("${OPENIMAGEIO_INCLUDES}")
endif ()


checked_find_package (PugiXML REQUIRED)


# LLVM library setup
checked_find_package (LLVM 6.0 REQUIRED
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


checked_find_package (partio)


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
        message (FATAL_ERROR "NVTPX target is not available in the provided LLVM build")
    endif()

    if (${CUDA_VERSION} VERSION_GREATER 8 AND ${LLVM_VERSION} VERSION_LESS 6)
        message (FATAL_ERROR "CUDA ${CUDA_VERSION} requires LLVM 6.0 or greater")
    endif ()

    set (CUDA_LIB_FLAGS "--cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")
endif ()


# OptiX setup
if (USE_OPTIX)
    checked_find_package (OptiX 5.1 REQUIRED)
    include_directories (BEFORE "${OPTIX_INCLUDES}")
    if (NOT USE_LLVM_BITCODE OR NOT USE_FAST_MATH)
        message (FATAL_ERROR "Enabling OptiX requires USE_LLVM_BITCODE=1 and USE_FAST_MATH=1")
    endif ()
endif ()
