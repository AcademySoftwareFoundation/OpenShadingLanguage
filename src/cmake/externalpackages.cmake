###########################################################################
# Find libraries

setup_path (THIRD_PARTY_TOOLS_HOME 
#            "${PROJECT_SOURCE_DIR}/../../external/dist/${platform}"
            "unknown"
            "Location of third party libraries in the external project")

# Add all third party tool directories to the include and library paths so
# that they'll be correctly found by the various FIND_PACKAGE() invocations.
if (THIRD_PARTY_TOOLS_HOME AND EXISTS ${THIRD_PARTY_TOOLS_HOME})
    set (CMAKE_INCLUDE_PATH "${THIRD_PARTY_TOOLS_HOME}/include" ${CMAKE_INCLUDE_PATH})
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

# example of using setup_var instead:
#setup_var (ILMBASE_VERSION 1.0.1 "Version of the ILMBase library")
setup_string (ILMBASE_VERSION 1.0.1
              "Version of the ILMBase library")
mark_as_advanced (ILMBASE_VERSION)
setup_path (ILMBASE_HOME "${THIRD_PARTY_TOOLS_HOME}"
            "Location of the ILMBase library install")
mark_as_advanced (ILMBASE_HOME)

find_package (IlmBase REQUIRED)

if (ILMBASE_FOUND)
    include_directories ("${ILMBASE_INCLUDE_DIR}")
    include_directories ("${ILMBASE_INCLUDE_DIR}/OpenEXR")
    if (VERBOSE)
        message (STATUS "ILMBASE_INCLUDE_DIR=${ILMBASE_INCLUDE_DIR}")
    endif ()
else ()
    message (STATUS "ILMBASE not found!")
endif ()

macro (LINK_ILMBASE target)
    target_link_libraries (${target} ${ILMBASE_LIBRARIES})
endmacro ()

# end IlmBase setup
###########################################################################


###########################################################################
# Boost setup

set (Boost_ADDITIONAL_VERSIONS "1.52" "1.51" "1.50" "1.49" "1.48"
                               "1.47" "1.46" "1.45"
                               "1.44" "1.43" "1.42" "1.41" "1.40")
if (LINKSTATIC)
    set (Boost_USE_STATIC_LIBS   ON)
endif ()
set (Boost_USE_MULTITHREADED ON)
if (BOOST_CUSTOM)
    set (Boost_FOUND true)
else ()
    set (Boost_COMPONENTS filesystem regex system thread)
    if (USE_BOOST_WAVE)
        list (APPEND Boost_COMPONENTS wave)
    endif ()

    find_package (Boost 1.40 REQUIRED 
                  COMPONENTS ${Boost_COMPONENTS}
                 )
endif ()

if (VERBOSE)
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
    find_library (PARTIO_LIBRARIES
                  NAMES partio
                  PATHS ${PARTIO_HOME}/lib)
    find_path (PARTIO_INCLUDE_DIR
               NAMES Partio.h
               PATHS ${PARTIO_HOME}/include)
    if (PARTIO_INCLUDE_DIR AND PARTIO_LIBRARIES)
        set (PARTIO_FOUND TRUE)
        add_definitions ("-DUSE_PARTIO=1")
        include_directories ("${PARTIO_INCLUDE_DIR}")
        if (VERBOSE)
            message (STATUS "Partio include = ${PARTIO_INCLUDE_DIR}")
            message (STATUS "Partio library = ${PARTIO_LIBRARIES}")
        endif ()
    else ()
        add_definitions ("-DUSE_PARTIO=0")
        set (PARTIO_FOUND FALSE)
        set (PARTIO_LIBRARIES "")
        message (STATUS "Partio not found")
    endif ()
else ()
    set (PARTIO_FOUND FALSE)
    set (PARTIO_LIBRARIES "")
endif (USE_PARTIO)

# end GL Extension Wrangler library setup
###########################################################################


###########################################################################
# LLVM library setup

# try to find llvm-config, with a specific version if specified
if(LLVM_DIRECTORY)
  FIND_PROGRAM(LLVM_CONFIG llvm-config-${LLVM_VERSION} HINTS ${LLVM_DIRECTORY}/bin NO_CMAKE_PATH)
  if(NOT LLVM_CONFIG)
    FIND_PROGRAM(LLVM_CONFIG llvm-config HINTS ${LLVM_DIRECTORY}/bin NO_CMAKE_PATH)
  endif()
else()
  FIND_PROGRAM(LLVM_CONFIG llvm-config-${LLVM_VERSION})
  if(NOT LLVM_CONFIG)
    FIND_PROGRAM(LLVM_CONFIG llvm-config)
  endif()
endif()

if(NOT LLVM_DIRECTORY OR EXISTS ${LLVM_CONFIG})
  execute_process (COMMAND ${LLVM_CONFIG} --version
       OUTPUT_VARIABLE LLVM_VERSION
       OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process (COMMAND ${LLVM_CONFIG} --prefix
       OUTPUT_VARIABLE LLVM_DIRECTORY
       OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process (COMMAND ${LLVM_CONFIG} --libdir
       OUTPUT_VARIABLE LLVM_LIB_DIR
       OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process (COMMAND ${LLVM_CONFIG} --includedir
       OUTPUT_VARIABLE LLVM_INCLUDES
       OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

find_library ( LLVM_LIBRARY
               NAMES LLVM-${LLVM_VERSION}
               PATHS ${LLVM_LIB_DIR})
if (VERBOSE)
    message (STATUS "LLVM version  = ${LLVM_VERSION}")
    message (STATUS "LLVM dir      = ${LLVM_DIRECTORY}")
    message (STATUS "LLVM includes = ${LLVM_INCLUDES}")
    message (STATUS "LLVM library  = ${LLVM_LIBRARY}")
    message (STATUS "LLVM lib dir  = ${LLVM_LIB_DIR}")
endif ()

# shared llvm library may not be available, this is not an error if we use LLVM_STATIC.
if ((LLVM_LIBRARY OR LLVM_STATIC) AND LLVM_INCLUDES AND LLVM_DIRECTORY AND LLVM_LIB_DIR)
  # ensure include directory is added (in case of non-standard locations
  include_directories (BEFORE "${LLVM_INCLUDES}")
  # Extract any wayward dots or "svn" suffixes from the version to yield
  # an integer version number we can use to make compilation decisions.
  string (REGEX REPLACE "\\." "" OSL_LLVM_VERSION ${LLVM_VERSION})
  string (REGEX REPLACE "svn" "" OSL_LLVM_VERSION ${OSL_LLVM_VERSION})
  add_definitions ("-DOSL_LLVM_VERSION=${OSL_LLVM_VERSION}")
  if (LLVM_STATIC)
    # if static LLVM libraries were requested, use llvm-config to generate
    # the list of what libraries we need, and substitute that in the right
    # way for LLVM_LIBRARY.
    execute_process (COMMAND ${LLVM_CONFIG} --libfiles
                     OUTPUT_VARIABLE LLVM_LIBRARY
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    string (REPLACE " " ";" LLVM_LIBRARY ${LLVM_LIBRARY})
  endif ()
  if (VERBOSE)
      message (STATUS "LLVM OSL_LLVM_VERSION = ${OSL_LLVM_VERSION}")
      message (STATUS "LLVM library  = ${LLVM_LIBRARY}")
  endif ()


  if (NOT LLVM_LIBRARY)
    message (FATAL_ERROR "LLVM library not found.")
  endif()
else ()
  message (FATAL_ERROR "LLVM not found.")
endif ()

# end LLVM library setup
###########################################################################
