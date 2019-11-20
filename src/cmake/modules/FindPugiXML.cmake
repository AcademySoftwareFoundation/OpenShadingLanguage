# Find the pugixml XML parsing library.
#
# Sets the usual variables expected for find_package scripts:
#
# PUGIXML_INCLUDES - header location
# PUGIXML_LIBRARIES - library to link against
# PUGIXML_FOUND - true if pugixml was found.

find_path (PUGIXML_INCLUDE_DIR
           NAMES pugixml.hpp
           HINTS /usr/local/include/pugixml-1.8)
find_library (PUGIXML_LIBRARY
              NAMES pugixml
              HINTS /usr/local/lib/pugixml-1.8)

# Second chance -- if not found, look in the OIIO distro
if (NOT PUGIXML_INCLUDE_DIR AND OPENIMAGEIO_INCLUDES)
    find_path (PUGIXML_INCLUDE_DIR
               NAMES pugixml.hpp
               HINTS ${OPENIMAGEIO_INCLUDES}
               PATH_SUFFIXES OpenImageIO)
    set (PUGIXML_LIBRARY ${OPENIMAGEIO_LIBRARIES})
endif ()


# Support the REQUIRED and QUIET arguments, and set PUGIXML_FOUND if found.
include (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS (PugiXML DEFAULT_MSG PUGIXML_LIBRARY
                                   PUGIXML_INCLUDE_DIR)

if (PUGIXML_FOUND)
    set (PUGIXML_INCLUDES ${PUGIXML_INCLUDE_DIR})
    set (PUGIXML_LIBRARIES ${PUGIXML_LIBRARY})
endif()

mark_as_advanced (PUGIXML_LIBRARY PUGIXML_INCLUDE_DIR)
