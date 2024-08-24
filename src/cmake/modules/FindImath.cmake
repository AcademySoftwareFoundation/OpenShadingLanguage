# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Module to find Imath.
#
# For Imath 3.x, this will establish the following imported
# targets:
#
#    Imath::Imath
#    Imath::Half
#
# and sets the following CMake variables:
#
#   IMATH_FOUND            true, if found
#   IMATH_VERSION          Imath version
#   IMATH_INCLUDES         directory where Imath headers are found
#

# First, try to fine just the right config files
find_package(Imath CONFIG)

if (TARGET Imath::Imath)
    # Imath 3.x if both of these targets are found
    if (NOT Imath_FIND_QUIETLY)
        message (STATUS "Found CONFIG for Imath 3 (Imath_VERSION=${Imath_VERSION})")
    endif ()

    # Mimic old style variables
    set (IMATH_VERSION ${Imath_VERSION})
    get_target_property(IMATH_INCLUDES Imath::Imath INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(IMATH_LIBRARY Imath::Imath INTERFACE_LINK_LIBRARIES)
    set (IMATH_LIBRARIES ${IMATH_LIBRARY})
    set (IMATH_FOUND true)

endif ()
