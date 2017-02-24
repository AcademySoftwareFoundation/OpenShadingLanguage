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


setup_path (THIRD_PARTY_TOOLS_HOME
            "unknown"
            "Location of third party libraries in the external project")

# Add all third party tool directories to the include and library paths so
# that they'll be correctly found by the various FIND_PACKAGE() invocations.
if (THIRD_PARTY_TOOLS_HOME AND EXISTS "${THIRD_PARTY_TOOLS_HOME}")
    set (CMAKE_INCLUDE_PATH "${THIRD_PARTY_TOOLS_HOME}/include" "${CMAKE_INCLUDE_PATH}")
    # Detect third party tools which have been successfully built using the
    # lock files which are placed there by the external project Makefile.
    file (GLOB _external_dir_lockfiles "${THIRD_PARTY_TOOLS_HOME}/*.d")
    foreach (_dir_lockfile ${_external_dir_lockfiles})
        # Grab the tool directory_name.d
        get_filename_component (_ext_dirname ${_dir_lockfile} NAME)
        # Strip off the .d extension
        string (REGEX REPLACE "\\.d$" "" _ext_dirname ${_ext_dirname})
        set (CMAKE_INCLUDE_PATH "${THIRD_PARTY_TOOLS_HOME}/include/${_ext_dirname}" ${CMAKE_INCLUDE_PATH})
        set (CMAKE_LIBRARY_PATH "${THIRD_PARTY_TOOLS_HOME}/lib/${_ext_dirname}" ${CMAKE_LIBRARY_PATH})
    endforeach ()
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
# OpenImageIO

find_package (OpenImageIO 1.6 REQUIRED)
include_directories ("${OPENIMAGEIO_INCLUDE_DIR}")
link_directories ("${OPENIMAGEIO_LIBRARY_DIRS}")
message (STATUS "Using OpenImageIO ${OPENIMAGEIO_VERSION}")

# end OpenImageIO setup
###########################################################################


###########################################################################
# Boost setup

if (NOT Boost_FIND_QUIETLY)
    message (STATUS "BOOST_ROOT ${BOOST_ROOT}")
endif ()

if (NOT DEFINED Boost_ADDITIONAL_VERSIONS)
  set (Boost_ADDITIONAL_VERSIONS "1.60" "1.59" "1.58" "1.57" "1.56"
                                 "1.55" "1.54" "1.53" "1.52" "1.51" "1.50"
                                 "1.49" "1.48" "1.47" "1.46" "1.45" "1.44"
                                 "1.43" "1.43.0" "1.42" "1.42.0")
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
    find_package (Boost 1.42 REQUIRED
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

# end GL Extension Wrangler library setup
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

# try to find llvm-config, with a specific version if specified
if (LLVM_DIRECTORY)
    set (LLVM_CONFIG_PATH_HINTS "${LLVM_DIRECTORY}/bin")
endif ()
list (APPEND LLVM_CONFIG_PATH_HINTS
        "/usr/local/opt/llvm/${LLVM_VERSION}/bin/"
        "/usr/local/opt/llvm/bin/")
find_program (LLVM_CONFIG
              NAMES llvm-config-${LLVM_VERSION} llvm-config
              HINTS ${LLVM_CONFIG_PATH_HINTS} NO_DEFAULT_PATH)
find_program (LLVM_CONFIG
              NAMES llvm-config-${LLVM_VERSION} llvm-config
              HINTS ${LLVM_CONFIG_PATH_HINTS})
if (VERBOSE)
    message (STATUS "Found llvm-config '${LLVM_CONFIG}'")
endif ()

if (NOT LLVM_DIRECTORY)
    execute_process (COMMAND ${LLVM_CONFIG} --prefix
           OUTPUT_VARIABLE LLVM_DIRECTORY
           OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

execute_process (COMMAND ${LLVM_CONFIG} --version
       OUTPUT_VARIABLE LLVM_VERSION
       OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process (COMMAND ${LLVM_CONFIG} --libdir
       OUTPUT_VARIABLE LLVM_LIB_DIR
       OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process (COMMAND ${LLVM_CONFIG} --includedir
       OUTPUT_VARIABLE LLVM_INCLUDES
       OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT ${LLVM_VERSION} VERSION_LESS 3.8)
    execute_process (COMMAND ${LLVM_CONFIG} --system-libs
                     OUTPUT_VARIABLE LLVM_SYSTEM_LIBRARIES
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
else ()
    # Older LLVM did not have llvm-config --system-libs, but we know that
    # on Linux, we'll need curses.
    find_package (Curses)
    if (CURSES_FOUND)
        list (APPEND LLVM_SYSTEM_LIBRARIES ${CURSES_LIBRARIES})
    endif ()
endif ()

find_library ( LLVM_LIBRARY
               NAMES LLVM-${LLVM_VERSION} LLVM
               PATHS ${LLVM_LIB_DIR})
find_library ( LLVM_MCJIT_LIBRARY
               NAMES LLVMMCJIT
               PATHS ${LLVM_LIB_DIR})

# if (NOT LLVM_LIBRARY)
#     execute_process (COMMAND ${LLVM_CONFIG} --libfiles engine
#                      OUTPUT_VARIABLE LLVM_LIBRARIES
#                      OUTPUT_STRIP_TRAILING_WHITESPACE)
# endif ()

# shared llvm library may not be available, this is not an error if we use LLVM_STATIC.
if ((LLVM_LIBRARY OR LLVM_LIBRARIES OR LLVM_STATIC) AND LLVM_INCLUDES AND LLVM_DIRECTORY AND LLVM_LIB_DIR)
  # ensure include directory is added (in case of non-standard locations
  include_directories (BEFORE "${LLVM_INCLUDES}")
  if (NOT OSL_LLVM_VERSION)
      # Extract and concatenate major & minor, remove wayward patches,
      # dots, and "svn" or other suffixes.
      string (REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1\\2" OSL_LLVM_VERSION ${LLVM_VERSION})
  endif ()
  add_definitions (-DOSL_LLVM_VERSION=${OSL_LLVM_VERSION})
  add_definitions (-DOSL_LLVM_FULL_VERSION="${LLVM_VERSION}")
  link_directories ("${LLVM_LIB_DIR}")
  if (LLVM_STATIC)
    # if static LLVM libraries were requested, use llvm-config to generate
    # the list of what libraries we need, and substitute that in the right
    # way for LLVM_LIBRARY.
    execute_process (COMMAND ${LLVM_CONFIG} --libfiles
                     OUTPUT_VARIABLE LLVM_LIBRARIES
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    string (REPLACE " " ";" LLVM_LIBRARIES "${LLVM_LIBRARIES}")
    set (LLVM_LIBRARY "")
  else ()
    set (LLVM_LIBRARIES "${LLVM_LIBRARY}")
  endif ()
endif ()

message (STATUS "LLVM version  = ${LLVM_VERSION}")
if (NOT LLVM_FIND_QUIETLY)
    message (STATUS "LLVM OSL_LLVM_VERSION = ${OSL_LLVM_VERSION}")
    message (STATUS "LLVM dir       = ${LLVM_DIRECTORY}")
    message (STATUS "LLVM includes  = ${LLVM_INCLUDES}")
    message (STATUS "LLVM lib dir   = ${LLVM_LIB_DIR}")
    message (STATUS "LLVM libraries = ${LLVM_LIBRARIES}")
    message (STATUS "LLVM sys libs  = ${LLVM_SYSTEM_LIBRARIES}")
endif ()

if (NOT LLVM_LIBRARIES)
    message (FATAL_ERROR "LLVM not found.")
endif()

# end LLVM library setup
###########################################################################
