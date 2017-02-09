###########################################################################
# Find libraries

# When not in VERBOSE mode, try to make things as quiet as possible
if (NOT VERBOSE)
    set (Bison_FIND_QUIETLY true)
    set (Boost_FIND_QUIETLY true)
    set (Curses_FIND_QUIETLY true)
    set (Flex_FIND_QUIETLY true)
    set (LLVM_FIND_QUIETLY true)
    set (OpenEXR_FIND_QUIETLY true)
    set (OpenImageIO_FIND_QUIETLY true)
    set (Partio_FIND_QUIETLY true)
    set (PkgConfig_FIND_QUIETLY true)
    set (PugiXML_FIND_QUIETLY TRUE)
    set (PythonInterp_FIND_QUIETLY true)
    set (PythonLibs_FIND_QUIETLY true)
    set (Threads_FIND_QUIETLY true)
    set (ZLIB_FIND_QUIETLY true)
endif ()


setup_string (SPECIAL_COMPILE_FLAGS ""
               "Custom compilation flags")
if (SPECIAL_COMPILE_FLAGS)
    add_definitions (${SPECIAL_COMPILE_FLAGS})
endif ()



###########################################################################
# IlmBase setup

find_package (OpenEXR REQUIRED)
#OpenEXR 2.2 still has problems with importing ImathInt64.h unqualified
#thus need for ilmbase/OpenEXR
include_directories ("${OPENEXR_INCLUDE_DIR}"
                     "${ILMBASE_INCLUDE_DIR}"
                     "${ILMBASE_INCLUDE_DIR}/OpenEXR")
if (${OPENEXR_VERSION} VERSION_LESS 2.0.0)
    # OpenEXR 1.x had weird #include dirctives, this is also necessary:
    include_directories ("${OPENEXR_INCLUDE_DIR}/OpenEXR")
else ()
    add_definitions (-DUSE_OPENEXR_VERSION2=1)
endif ()
if (NOT OpenEXR_FIND_QUIETLY)
    message (STATUS "ILMBASE_INCLUDE_DIR = ${ILMBASE_INCLUDE_DIR}")
    message (STATUS "ILMBASE_LIBRARIES = ${ILMBASE_LIBRARIES}")
endif ()

# end IlmBase setup
###########################################################################


###########################################################################
# Boost setup

if (NOT Boost_FIND_QUIETLY)
    message (STATUS "BOOST_ROOT ${BOOST_ROOT}")
endif ()

if (NOT DEFINED Boost_ADDITIONAL_VERSIONS)
  set (Boost_ADDITIONAL_VERSIONS "1.63" "1.62" "1.61" "1.60"
                                 "1.59" "1.58" "1.57" "1.56" "1.55")
endif ()
if (LINKSTATIC)
    set (Boost_USE_STATIC_LIBS   ON)
endif ()
set (Boost_USE_MULTITHREADED ON)
if (BOOST_CUSTOM)
    set (Boost_FOUND true)
    # N.B. For a custom version, the caller had better set up the variables
    # Boost_VERSION, Boost_INCLUDE_DIRS, Boost_LIBRARY_DIRS, Boost_LIBRARIES.
else ()
    set (Boost_COMPONENTS filesystem regex system thread wave)
    find_package (Boost 1.55 REQUIRED
                  COMPONENTS ${Boost_COMPONENTS}
                 )
endif ()

# On Linux, Boost 1.55 and higher seems to need to link against -lrt
if (CMAKE_SYSTEM_NAME MATCHES "Linux" AND ${Boost_VERSION} GREATER 105499)
    list (APPEND Boost_LIBRARIES "rt")
endif ()

if (NOT Boost_FIND_QUIETLY)
    message (STATUS "BOOST_ROOT ${BOOST_ROOT}")
    message (STATUS "Boost found ${Boost_FOUND} ")
    message (STATUS "Boost version      ${Boost_VERSION}")
    message (STATUS "Boost include dirs ${Boost_INCLUDE_DIRS}")
    message (STATUS "Boost library dirs ${Boost_LIBRARY_DIRS}")
    message (STATUS "Boost libraries    ${Boost_LIBRARIES}")
endif ()

include_directories (SYSTEM "${Boost_INCLUDE_DIRS}")
link_directories ("${Boost_LIBRARY_DIRS}")

# end Boost setup
###########################################################################


###########################################################################
# OpenImageIO

find_package (OpenImageIO 1.7 REQUIRED)
include_directories ("${OPENIMAGEIO_INCLUDE_DIR}")
message (STATUS "Using OpenImageIO ${OPENIMAGEIO_VERSION}")

# end OpenImageIO setup
###########################################################################


###########################################################################
# Partio

find_package (ZLIB)
if (USE_PARTIO)
    find_package (Partio)
    if (PARTIO_FOUND)
        add_definitions ("-DUSE_PARTIO=1")
        include_directories ("${PARTIO_INCLUDE_DIR}")
    else ()
        add_definitions ("-DUSE_PARTIO=0")
    endif ()
endif (USE_PARTIO)

# end Partio setup
###########################################################################


###########################################################################
# Pugixml setup.  Normally we just use the version bundled with oiio, but
# some linux distros are quite particular about having separate packages so we
# allow this to be overridden to use the distro-provided package if desired.
if (USE_EXTERNAL_PUGIXML)
    find_package (PugiXML REQUIRED)
    # insert include path to pugixml first, to ensure that the external
    # pugixml is found, and not the one in OIIO's include directory.
    include_directories (BEFORE "${PUGIXML_INCLUDE_DIR}")
endif()
# end Pugixml setup
###########################################################################


###########################################################################
# LLVM library setup

find_package (LLVM 3.4 REQUIRED)

if (LLVM_FOUND)
  # ensure include directory is added (in case of non-standard locations
  include_directories (BEFORE "${LLVM_INCLUDES}")
  link_directories ("${LLVM_LIB_DIR}")
  # Extract and concatenate major & minor, remove wayward patches,
  # dots, and "svn" or other suffixes.
  string (REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1\\2" OSL_LLVM_VERSION ${LLVM_VERSION})
  add_definitions (-DOSL_LLVM_VERSION=${OSL_LLVM_VERSION})
  add_definitions (-DOSL_LLVM_FULL_VERSION="${LLVM_VERSION}")
endif ()

# end LLVM library setup
###########################################################################
