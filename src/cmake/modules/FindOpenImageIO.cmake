###########################################################################
# CMake module to find OpenImageIO
#
# This module will set
#   OPENIMAGEIO_FOUND          True, if found
#   OPENIMAGEIO_INCLUDE_DIR    directory where headers are found
#   OPENIMAGEIO_LIBRARIES      libraries for OIIO
#   OPENIMAGEIO_LIBRARY_DIRS   library dirs for OIIO
#   OPENIMAGEIO_VERSION        Version ("major.minor.patch")
#   OPENIMAGEIO_VERSION_MAJOR  Version major number
#   OPENIMAGEIO_VERSION_MINOR  Version minor number
#   OPENIMAGEIO_VERSION_PATCH  Version minor patch
#
# Special inputs:
#   OPENIMAGEIOHOME - custom "prefix" location of OIIO installation
#                      (expecting bin, lib, include subdirectories)
#


# If 'OPENIMAGEHOME' not set, use the env variable of that name if available
if (NOT OPENIMAGEIOHOME AND NOT $ENV{OPENIMAGEIOHOME} STREQUAL "")
    set (OPENIMAGEIOHOME $ENV{OPENIMAGEIOHOME})
endif ()


if (NOT OpenImageIO_FIND_QUIETLY)
    message ( STATUS "OPENIMAGEIOHOME = ${OPENIMAGEIOHOME}" )
endif ()

find_library ( OPENIMAGEIO_LIBRARY
               NAMES OpenImageIO
               HINTS ${OPENIMAGEIOHOME}/lib
               PATH_SUFFIXES lib64 lib
               PATHS "${OPENIMAGEIOHOME}/lib" )
find_path ( OPENIMAGEIO_INCLUDE_DIR
            NAMES OpenImageIO/imageio.h
            HINTS ${OPENIMAGEIOHOME}/include
            PATH_SUFFIXES include )
find_program ( OPENIMAGEIO_BIN
               NAMES oiiotool
               HINTS ${OPENIMAGEIOHOME}/bin
               PATH_SUFFIXES bin )

# Try to figure out version number
set (OIIO_VERSION_HEADER "${OPENIMAGEIO_INCLUDE_DIR}/OpenImageIO/oiioversion.h")
if (EXISTS "${OIIO_VERSION_HEADER}")
    file (STRINGS "${OIIO_VERSION_HEADER}" TMP REGEX "^#define OIIO_VERSION_MAJOR .*$")
    string (REGEX MATCHALL "[0-9]+" OPENIMAGEIO_VERSION_MAJOR ${TMP})
    file (STRINGS "${OIIO_VERSION_HEADER}" TMP REGEX "^#define OIIO_VERSION_MINOR .*$")
    string (REGEX MATCHALL "[0-9]+" OPENIMAGEIO_VERSION_MINOR ${TMP})
    file (STRINGS "${OIIO_VERSION_HEADER}" TMP REGEX "^#define OIIO_VERSION_PATCH .*$")
    string (REGEX MATCHALL "[0-9]+" OPENIMAGEIO_VERSION_PATCH ${TMP})
    set (OPENIMAGEIO_VERSION "${OPENIMAGEIO_VERSION_MAJOR}.${OPENIMAGEIO_VERSION_MINOR}.${OPENIMAGEIO_VERSION_PATCH}")
endif ()

set ( OPENIMAGEIO_LIBRARIES ${OPENIMAGEIO_LIBRARY})
get_filename_component (OPENIMAGEIO_LIBRARY_DIRS "${OPENIMAGEIO_LIBRARY}" DIRECTORY CACHE)

if (NOT OpenImageIO_FIND_QUIETLY)
    message ( STATUS "OpenImageIO includes     = ${OPENIMAGEIO_INCLUDE_DIR}" )
    message ( STATUS "OpenImageIO libraries    = ${OPENIMAGEIO_LIBRARIES}" )
    message ( STATUS "OpenImageIO library_dirs = ${OPENIMAGEIO_LIBRARY_DIRS}" )
    message ( STATUS "OpenImageIO bin          = ${OPENIMAGEIO_BIN}" )
endif ()

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (OpenImageIO
    FOUND_VAR     OPENIMAGEIO_FOUND
    REQUIRED_VARS OPENIMAGEIO_INCLUDE_DIR OPENIMAGEIO_LIBRARIES
                  OPENIMAGEIO_LIBRARY_DIRS OPENIMAGEIO_VERSION
    VERSION_VAR   OPENIMAGEIO_VERSION
    )

mark_as_advanced (
    OPENIMAGEIO_INCLUDE_DIR
    OPENIMAGEIO_LIBRARIES
    OPENIMAGEIO_LIBRARY_DIRS
    OPENIMAGEIO_VERSION
    OPENIMAGEIO_VERSION_MAJOR
    OPENIMAGEIO_VERSION_MINOR
    OPENIMAGEIO_VERSION_PATCH
    )
