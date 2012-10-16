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
    message (STATUS "ILMBASE_INCLUDE_DIR=${ILMBASE_INCLUDE_DIR}")
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

message (STATUS "BOOST_ROOT ${BOOST_ROOT}")

set (Boost_ADDITIONAL_VERSIONS "1.51" "1.50" "1.49" "1.48"
                               "1.47" "1.46" "1.45"
                               "1.44" "1.43" "1.42" "1.41" "1.40")
#set (Boost_USE_STATIC_LIBS   ON)
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

message (STATUS "Boost found ${Boost_FOUND} ")
message (STATUS "Boost include dirs ${Boost_INCLUDE_DIRS}")
message (STATUS "Boost library dirs ${Boost_LIBRARY_DIRS}")
message (STATUS "Boost libraries    ${Boost_LIBRARIES}")

include_directories (SYSTEM "${Boost_INCLUDE_DIRS}")
link_directories ("${Boost_LIBRARY_DIRS}")

# end Boost setup
###########################################################################


###########################################################################
# TBB (Intel Thread Building Blocks) setup

setup_path (TBB_HOME "${THIRD_PARTY_TOOLS_HOME}"
            "Location of the TBB library install")
mark_as_advanced (TBB_HOME)
if (USE_TBB)
    set (TBB_VERSION 22_004oss)
    if (MSVC)
        find_library (TBB_LIBRARY
                      NAMES tbb
                      PATHS ${TBB_HOME}/lib
                      PATHS ${THIRD_PARTY_TOOLS_HOME}/lib/
                      ${TBB_HOME}/tbb-${TBB_VERSION}/lib/
                     )
        find_library (TBB_DEBUG_LIBRARY
                      NAMES tbb_debug
                      PATHS ${TBB_HOME}/lib
                      PATHS ${THIRD_PARTY_TOOLS_HOME}/lib/
                      ${TBB_HOME}/tbb-${TBB_VERSION}/lib/)
    endif (MSVC)
    find_path (TBB_INCLUDES tbb/tbb_stddef.h
               ${TBB_HOME}/include/tbb${TBB_VERSION}
               ${THIRD_PARTY_TOOLS}/include/tbb${TBB_VERSION}
               ${PROJECT_SOURCE_DIR}/include
               ${OPENIMAGEIOHOME}/include/OpenImageIO
              )
    if (TBB_INCLUDES OR TBB_LIBRARY)
        set (TBB_FOUND TRUE)
        message (STATUS "TBB includes = ${TBB_INCLUDES}")
        message (STATUS "TBB library = ${TBB_LIBRARY}")
        add_definitions ("-DUSE_TBB=1")
    else ()
        message (STATUS "TBB not found")
    endif ()
else ()
    add_definitions ("-DUSE_TBB=0")
    message (STATUS "TBB will not be used")
    set(TBB_INCLUDES "")
    set(TBB_LIBRARY "")
endif ()

# end TBB setup
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
        message (STATUS "Partio include = ${PARTIO_INCLUDE_DIR}")
        message (STATUS "Partio library = ${PARTIO_LIBRARIES}")
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

if (LLVM_DIRECTORY)
    set (LLVM_CONFIG "${LLVM_DIRECTORY}/bin/llvm-config")
else ()
    set (LLVM_CONFIG llvm-config)
endif ()
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
find_library ( LLVM_LIBRARY
               NAMES LLVM-${LLVM_VERSION}
               PATHS ${LLVM_LIB_DIR})
message (STATUS "LLVM version  = ${LLVM_VERSION}")
message (STATUS "LLVM dir      = ${LLVM_DIRECTORY}")
message (STATUS "LLVM includes = ${LLVM_INCLUDES}")
message (STATUS "LLVM library  = ${LLVM_LIBRARY}")
message (STATUS "LLVM lib dir  = ${LLVM_LIB_DIR}")

if (LLVM_LIBRARY AND LLVM_INCLUDES AND LLVM_DIRECTORY AND LLVM_LIB_DIR)
  # ensure include directory is added (in case of non-standard locations
  include_directories (BEFORE "${LLVM_INCLUDES}")
  # Extract any wayward dots or "svn" suffixes from the version to yield
  # an integer version number we can use to make compilation decisions.
  string (REGEX REPLACE "\\." "" OSL_LLVM_VERSION ${LLVM_VERSION})
  string (REGEX REPLACE "svn" "" OSL_LLVM_VERSION ${OSL_LLVM_VERSION})
  message (STATUS "LLVM OSL_LLVM_VERSION = ${OSL_LLVM_VERSION}")
  add_definitions ("-DOSL_LLVM_VERSION=${OSL_LLVM_VERSION}")
  if (LLVM_STATIC)
    # if static LLVM libraries were requested, use llvm-config to generate
    # the list of what libraries we need, and substitute that in the right
    # way for LLVM_LIBRARY.
    set (LLVM_LIBRARY "")
    execute_process (COMMAND ${LLVM_CONFIG} --libs
                 OUTPUT_VARIABLE llvm_library_list
	         OUTPUT_STRIP_TRAILING_WHITESPACE)
    string (REPLACE "-l" "" llvm_library_list ${llvm_library_list})
    string (REPLACE " " ";" llvm_library_list ${llvm_library_list})
    foreach (f ${llvm_library_list})
      list (APPEND LLVM_LIBRARY "${LLVM_LIB_DIR}/lib${f}.a")
    endforeach ()
  endif ()
  message (STATUS "LLVM library  = ${LLVM_LIBRARY}")
else ()
  message (FATAL_ERROR "LLVM not found.")
endif ()

# end LLVM library setup
###########################################################################
