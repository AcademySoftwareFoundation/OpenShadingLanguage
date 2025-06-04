# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

###########################################################################
# CMake module to find OptiX
#
# This module will set
#   OPTIX_FOUND          True, if OptiX is found
#   OPTIX_INCLUDES       directory where OptiX headers are found
#   OPTIX_LIBRARIES      libraries for OptiX
#
# Special inputs:
#   OPTIXHOME - custom "prefix" location of OptiX installation
#                       (expecting bin, lib, include subdirectories)


# If 'OPTIXHOME' not set, use the env variable of that name if available
if (NOT OPTIXHOME)
    if (NOT $ENV{OPTIXHOME} STREQUAL "")
        set (OPTIXHOME $ENV{OPTIXHOME})
    elseif (NOT $ENV{OPTIX_INSTALL_DIR} STREQUAL "")
        set (OPTIXHOME $ENV{OPTIX_INSTALL_DIR})
    endif ()
endif ()

if (NOT OptiX_FIND_QUIETLY)
    message (STATUS "OPTIXHOME = ${OPTIXHOME}")
endif ()

find_path (OPTIX_INCLUDE_DIR
    NAMES optix.h
    HINTS ${OPTIXHOME}/include
    PATH_SUFFIXES include )

if (OPTIX_INCLUDE_DIR)
    # Pull out the API version from optix.h
    file(STRINGS ${OPTIX_INCLUDE_DIR}/optix.h OPTIX_VERSION_LINE LIMIT_COUNT 1 REGEX "define OPTIX_VERSION")
    string(REGEX MATCH "([0-9]+)" OPTIX_VERSION_STRING "${OPTIX_VERSION_LINE}")
    math(EXPR OPTIX_VERSION_MAJOR "${OPTIX_VERSION_STRING}/10000")
    math(EXPR OPTIX_VERSION_MINOR "(${OPTIX_VERSION_STRING}%10000)/100")
    math(EXPR OPTIX_VERSION_MICRO "${OPTIX_VERSION_STRING}%100")
    set(OPTIX_VERSION "${OPTIX_VERSION_MAJOR}.${OPTIX_VERSION_MINOR}.${OPTIX_VERSION_MICRO}")
endif ()

mark_as_advanced (
    OPTIX_INCLUDE_DIR
    OPTIX_LIBRARIES
    OPTIX_VERSION
    )

include (FindPackageHandleStandardArgs)


find_package_handle_standard_args (OptiX
    FOUND_VAR     OPTIX_FOUND
    REQUIRED_VARS OPTIX_INCLUDE_DIR OPTIX_VERSION
    VERSION_VAR   OPTIX_VERSION
    )

if (OPTIX_FOUND)
    set (OPTIX_INCLUDES ${OPTIX_INCLUDE_DIR})
endif ()
