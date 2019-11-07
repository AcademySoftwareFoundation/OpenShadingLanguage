# Find the Partio library.
#
# Sets the usual variables expected for find_package scripts:
#
# PARTIO_INCLUDE_DIR - header location
# PARTIO_LIBRARIES - library to link against
# PARTIO_FOUND - true if Partio was found.

find_path (PARTIO_INCLUDE_DIR NAMES Partio.h)
find_library (PARTIO_LIBRARY NAMES partio)


# Support the REQUIRED and QUIET arguments, and set PARTIO_FOUND if found.
include (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS (Partio DEFAULT_MSG PARTIO_LIBRARY
                                   PARTIO_INCLUDE_DIR)

if (PARTIO_FOUND)
    set (PARTIO_INCLUDES ${PARTIO_INCLUDE_DIR})
    set (PARTIO_LIBRARIES ${PARTIO_LIBRARY})
endif()

