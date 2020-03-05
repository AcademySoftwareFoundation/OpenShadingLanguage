# Find the Partio library.
#
# Sets the usual variables expected for find_package scripts:
#
# PARTIO_INCLUDE_DIR - header location
# PARTIO_LIBRARIES - library to link against
# PARTIO_FOUND - true if Partio was found.

# Hack! perfer a config if it can be found
find_package(partio CONFIG)
if (PARTIO_FOUND OR partio_FOUND)
    if (VERBOSE)
        message (STATUS "partio found via config")
    endif ()
else ()
    if (VERBOSE)
        message (STATUS "partio falling back to FindPartio.cmake")
    endif ()


find_path (PARTIO_INCLUDE_DIR NAMES Partio.h)
find_library (PARTIO_LIBRARY NAMES partio)


# Support the REQUIRED and QUIET arguments, and set PARTIO_FOUND if found.
include (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS (partio DEFAULT_MSG PARTIO_LIBRARY
                                   PARTIO_INCLUDE_DIR)

if (PARTIO_FOUND)
    set (PARTIO_INCLUDES ${PARTIO_INCLUDE_DIR})
    set (PARTIO_LIBRARIES ${PARTIO_LIBRARY})
    if (NOT TARGET partio::partio)
        add_library (partio::partio UNKNOWN IMPORTED)
        set_target_properties (partio::partio PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${PARTIO_INCLUDE_DIR}")
        set_property (TARGET partio::partio APPEND PROPERTY
                      IMPORTED_LOCATION "${PARTIO_LIBRARY}")
    endif ()
endif()

endif ()
