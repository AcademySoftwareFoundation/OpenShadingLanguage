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
# LLVM library setup

find_package (LLVM 3.4 REQUIRED)

if (LLVM_FOUND)
  # ensure include directory is added (in case of non-standard locations
  include_directories (BEFORE SYSTEM "${LLVM_INCLUDES}")
  link_directories ("${LLVM_LIB_DIR}")
  # Extract and concatenate major & minor, remove wayward patches,
  # dots, and "svn" or other suffixes.
  string (REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1\\2" OSL_LLVM_VERSION ${LLVM_VERSION})
  add_definitions (-DOSL_LLVM_VERSION=${OSL_LLVM_VERSION})
  add_definitions (-DOSL_LLVM_FULL_VERSION="${LLVM_VERSION}")
endif ()

# end LLVM library setup
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
    if (USE_BOOST_WAVE)
        add_definitions (-DUSE_BOOST_WAVE=1)
    endif ()
else ()
    set (Boost_COMPONENTS system thread)
    if (NOT USE_STD_REGEX)
        list (APPEND Boost_COMPONENTS regex)
    endif ()
    if (CMAKE_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_APPLECLANG OR
            ${LLVM_VERSION} VERSION_LESS 3.6)
        set (_CLANG_PREPROCESSOR_CAN_WORK ON)
    endif ()
    if (GCC_VERSION)
        if (${GCC_VERSION} VERSION_LESS 4.9)
            set (_CLANG_PREPROCESSOR_CAN_WORK ON)
        endif ()
    endif ()
    if (${LLVM_VERSION} VERSION_LESS 3.9)
        # Bug in old LLVM creates some linkage problems we've seen involving
        # some singleton globals that are duplicated when we include both
        # the clang libs we need for the preprocessing as well as certain
        # LLVM support libraries we also need, triggering assertions.
        # See this for description of the issue:
        # http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20140203/203968.html
        # Sweep under the rug by falling back to boost wave when using older
        # LLVM (it seems fixed and no longer triggers for 3.9+).
        set (_CLANG_PREPROCESSOR_CAN_WORK OFF)
    endif ()
    if (USE_BOOST_WAVE OR (NOT CLANG_LIBRARIES)
        OR (NOT USE_CPP11 AND NOT USE_CPP14)
        OR (NOT _CLANG_PREPROCESSOR_CAN_WORK))
        # N.B. Using clang for preprocessing seems to work when using clang,
        # or gcc 4.8.x, or LLVM <= 3.5. When those conditions aren't met,
        # fall back on Boost Wave. We'll lift this restriction as soon as we
        # fix whatever is broken.
        # Also, for C++03, we need Boost Wave still, because we're too lazy
        # to deal with it.
        list (APPEND Boost_COMPONENTS filesystem wave)
        add_definitions (-DUSE_BOOST_WAVE=1)
        message (STATUS "Using Boost Wave for preprocessing")
    else ()
        message (STATUS "Using clang internals for preprocessing")
    endif ()
    find_package (Boost 1.42 REQUIRED
                  COMPONENTS ${Boost_COMPONENTS})
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
