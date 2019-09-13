# Module to find OpenColorIO
#
# This module will first look into the directories defined by the variables:
#   - OCIO_PATH, OCIO_LIBRARY_PATH
#
# This module defines the following variables:
#
# OCIO_FOUND       - True if OpenColorIO was found.
# OCIO_LIBRARIES   - list of libraries to link against when using OpenColorIO

# Other standard issue macros
include (FindPackageHandleStandardArgs)
include (FindPackageMessage)
message(STATUS "Looking for library: OpenColorIO")

if (NOT OpenColorIO_FIND_QUIETLY)
    if (OCIO_PATH)
        message(STATUS "OCIO path explicitly specified: ${OCIO_PATH}")
    endif()
    if (OCIO_LIBRARY_PATH)
        message(STATUS "OCIO LIBRARY_PATH explicitly specified: ${OCIO_LIBRARY_PATH}")
    endif()
endif ()
FIND_LIBRARY(OCIO_LIBRARIES
    NAMES OCIO OpenColorIO
    PATHS
    ${OCIO_LIBRARY_PATH}
    ${OCIO_PATH}/lib/
    /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
    /sw/lib
    /opt/local/lib
    DOC "The OCIO library")

if(OCIO_LIBRARIES)
    set(OCIO_FOUND TRUE)
    if (NOT OpenColorIO_FIND_QUIETLY)
        message(STATUS "Found OCIO library ${OCIO_LIBRARIES}")
    endif ()
else()
    set(OCIO_FOUND FALSE)
    message(STATUS "OCIO not found. Specify OCIO_PATH to locate it")
endif()

