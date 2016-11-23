# Find the Partio library.
#
# Sets the usual variables expected for find_package scripts:
#
# PARTIO_INCLUDE_DIR - header location
# PARTIO_LIBRARIES - library to link against
# PARTIO_FOUND - true if Partio was found.

find_library (PARTIO_LIBRARY
              NAMES partio
              PATHS "${PARTIO_HOME}/lib" "$ENV{PARTIO_HOME}/lib"
              NO_DEFAULT_PATH)
find_library (PARTIO_LIBRARY NAMES partio)
find_path (PARTIO_INCLUDE_DIR
           NAMES Partio.h
           PATHS "${PARTIO_HOME}/include" "$ENV{PARTIO_HOME}/include"
           NO_DEFAULT_PATH)
find_path (PARTIO_INCLUDE_DIR NAMES Partio.h)


# Support the REQUIRED and QUIET arguments, and set PARTIO_FOUND if found.
include (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS (Partio DEFAULT_MSG PARTIO_LIBRARY
                                   PARTIO_INCLUDE_DIR)

if (PARTIO_FOUND)
    set (PARTIO_LIBRARIES ${PARTIO_LIBRARY})
    if (NOT Partio_FIND_QUIETLY)
        message (STATUS "Partio include   = ${PARTIO_INCLUDE_DIR}")
        message (STATUS "Partio libraries = ${PARTIO_LIBRARIES}")
    endif ()
else ()
    message (STATUS "No Partio found")
endif()

mark_as_advanced (PARTIO_LIBRARIES PARTIO_INCLUDE_DIR)
