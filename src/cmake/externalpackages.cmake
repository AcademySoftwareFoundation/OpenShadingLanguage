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

find_package (OpenEXR 2.0 REQUIRED)
#OpenEXR 2.2 still has problems with importing ImathInt64.h unqualified
#thus need for ilmbase/OpenEXR
include_directories ("${OPENEXR_INCLUDE_DIR}"
                     "${ILMBASE_INCLUDE_DIR}"
                     "${ILMBASE_INCLUDE_DIR}/OpenEXR")
if (${OPENEXR_VERSION} VERSION_LESS 2.0.0)
    message (FATAL_ERROR "OpenEXR/Ilmbase is too old")
endif ()
if (NOT OpenEXR_FIND_QUIETLY)
    message (STATUS "OPENEXR_INCLUDE_DIR = ${OPENEXR_INCLUDE_DIR}")
    message (STATUS "OPENEXR_LIBRARIES = ${OPENEXR_LIBRARIES}")
endif ()

# end IlmBase setup
###########################################################################


###########################################################################
# OpenImageIO

find_package (OpenImageIO 1.7 REQUIRED)
include_directories ("${OPENIMAGEIO_INCLUDE_DIR}")
link_directories ("${OPENIMAGEIO_LIBRARY_DIRS}")
message (STATUS "Using OpenImageIO ${OPENIMAGEIO_VERSION}")

# end OpenImageIO setup
###########################################################################


###########################################################################
# LLVM library setup

find_package (LLVM 3.9 REQUIRED)

# ensure include directory is added (in case of non-standard locations
include_directories (BEFORE SYSTEM "${LLVM_INCLUDES}")
include_directories (BEFORE SYSTEM "${CLANG_INCLUDES}")
link_directories ("${LLVM_LIB_DIR}")
# Extract and concatenate major & minor, remove wayward patches,
# dots, and "svn" or other suffixes.
string (REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1\\2" OSL_LLVM_VERSION ${LLVM_VERSION})
add_definitions (-DOSL_LLVM_VERSION=${OSL_LLVM_VERSION})
add_definitions (-DOSL_LLVM_FULL_VERSION="${LLVM_VERSION}")
if (LLVM_NAMESPACE)
    add_definitions ("-DLLVM_NAMESPACE=${LLVM_NAMESPACE}")
endif ()

# end LLVM library setup
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
    set (Boost_USE_STATIC_LIBS ON)
endif ()
set (Boost_USE_MULTITHREADED ON)
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
if (MSVC)
    set (_CLANG_PREPROCESSOR_CAN_WORK ON)
endif ()
if (USE_BOOST_WAVE OR (NOT CLANG_LIBRARIES)
    OR (NOT _CLANG_PREPROCESSOR_CAN_WORK))
    # N.B. Using clang for preprocessing seems to work when using clang,
    # or gcc 4.8.x, or LLVM <= 3.5. When those conditions aren't met,
    # fall back on Boost Wave. We'll lift this restriction as soon as we
    # fix whatever is broken.
    list (APPEND Boost_COMPONENTS filesystem wave)
    add_definitions (-DUSE_BOOST_WAVE=1)
    message (STATUS "Using Boost Wave for preprocessing")
else ()
    message (STATUS "Using clang internals for preprocessing")
endif ()
if (BOOST_CUSTOM)
    set (Boost_FOUND true)
else ()
    find_package (Boost 1.55 REQUIRED
                  COMPONENTS ${Boost_COMPONENTS})
endif ()

# Needed for static boost libraries on Windows
if (WIN32 AND Boost_USE_STATIC_LIBS)
    add_definitions ("-DBOOST_ALL_NO_LIB")
    add_definitions ("-DBOOST_THREAD_USE_LIB")
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

# end Partio setup
###########################################################################


###########################################################################
# Pugixml setup.  Prefer a system install, but note that FindPugiXML.cmake
# will look in the OIIO distribution if it's not found on the system.
find_package (PugiXML REQUIRED)
include_directories (BEFORE "${PUGIXML_INCLUDE_DIR}")
# end Pugixml setup
###########################################################################
