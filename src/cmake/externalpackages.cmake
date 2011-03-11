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
# IlmBase and OpenEXR setup

# TODO: Place the OpenEXR stuff into a separate FindOpenEXR.cmake module.

# example of using setup_var instead:
#setup_var (ILMBASE_VERSION 1.0.1 "Version of the ILMBase library")
setup_string (ILMBASE_VERSION 1.0.1
              "Version of the ILMBase library")
mark_as_advanced (ILMBASE_VERSION)
setup_path (ILMBASE_HOME "${THIRD_PARTY_TOOLS_HOME}"
            "Location of the ILMBase library install")
mark_as_advanced (ILMBASE_HOME)
find_path (ILMBASE_INCLUDE_AREA OpenEXR/half.h
           ${ILMBASE_HOME}/include/ilmbase-${ILMBASE_VERSION}
           ${ILMBASE_HOME}/include
          )
if (ILMBASE_CUSTOM)
    set (IlmBase_Libraries ${ILMBASE_CUSTOM_LIBRARIES})
    separate_arguments(IlmBase_Libraries)
else ()
    set (IlmBase_Libraries Imath Half IlmThread Iex)
endif ()

foreach (_lib ${IlmBase_Libraries})
    find_library (current_lib ${_lib}
                  PATHS ${ILMBASE_HOME}/lib ${ILMBASE_HOME}/lib64
                        ${ILMBASE_LIB_AREA}
                  )
    list(APPEND ILMBASE_LIBRARIES ${current_lib})
    # this effectively unsets ${current_lib} so that find_library()
    # won't use the previous search results from the cache
    set (current_lib current_lib-NOTFOUND)
endforeach ()

message (STATUS "ILMBASE_INCLUDE_AREA = ${ILMBASE_INCLUDE_AREA}")
message (STATUS "ILMBASE_LIBRARIES = ${ILMBASE_LIBRARIES}")

if (ILMBASE_INCLUDE_AREA AND ILMBASE_LIBRARIES)
    set (ILMBASE_FOUND true)
    include_directories ("${ILMBASE_INCLUDE_AREA}")
    include_directories ("${ILMBASE_INCLUDE_AREA}/OpenEXR")
else ()
    message (FATAL_ERROR "ILMBASE not found!")
endif ()

macro (LINK_ILMBASE target)
    target_link_libraries (${target} ${ILMBASE_LIBRARIES})
endmacro ()

setup_string (OPENEXR_VERSION 1.6.1 "OpenEXR version number")
setup_string (OPENEXR_VERSION_DIGITS 010601 "OpenEXR version preprocessor number")
mark_as_advanced (OPENEXR_VERSION)
mark_as_advanced (OPENEXR_VERSION_DIGITS)
# FIXME -- should instead do the search & replace automatically, like this
# way it was done in the old makefiles:
#     OPENEXR_VERSION_DIGITS ?= 0$(subst .,0,${OPENEXR_VERSION})
setup_path (OPENEXR_HOME "${THIRD_PARTY_TOOLS_HOME}"
            "Location of the OpenEXR library install")
mark_as_advanced (OPENEXR_HOME)
find_path (OPENEXR_INCLUDE_AREA OpenEXR/OpenEXRConfig.h
           ${OPENEXR_HOME}/include
           ${ILMBASE_HOME}/include/openexr-${OPENEXR_VERSION}
          )
if (OPENEXR_CUSTOM)
    find_library (OPENEXR_LIBRARY ${OPENEXR_CUSTOM_LIBRARY}
                  PATHS ${OPENEXR_HOME}/lib
                        ${OPENEXR_HOME}/lib64
                        ${OPENEXR_LIB_AREA}
                 )
else ()
    find_library (OPENEXR_LIBRARY IlmImf
                  PATHS ${OPENEXR_HOME}/lib
                        ${OPENEXR_HOME}/lib64
                        ${OPENEXR_LIB_AREA}
                 )
endif ()

message (STATUS "OPENEXR_INCLUDE_AREA = ${OPENEXR_INCLUDE_AREA}")
message (STATUS "OPENEXR_LIBRARY = ${OPENEXR_LIBRARY}")
if (OPENEXR_INCLUDE_AREA AND OPENEXR_LIBRARY)
    set (OPENEXR_FOUND true)
    include_directories (${OPENEXR_INCLUDE_AREA})
    include_directories (${OPENEXR_INCLUDE_AREA}/OpenEXR)
else ()
    message (STATUS "OPENEXR not found!")
endif ()
add_definitions ("-DOPENEXR_VERSION=${OPENEXR_VERSION_DIGITS}")
find_package (ZLIB)
macro (LINK_OPENEXR target)
    target_link_libraries (${target} ${OPENEXR_LIBRARY} ${ZLIB_LIBRARIES})
endmacro ()


# end IlmBase and OpenEXR setup
###########################################################################

###########################################################################
# Boost setup

message (STATUS "BOOST_ROOT ${BOOST_ROOT}")

set (Boost_ADDITIONAL_VERSIONS "1.40" "1.39" "1.38" "1.38.0" "1.37" "1.37.0" "1.34.1" "1_34_1")
#set (Boost_USE_STATIC_LIBS   ON)
set (Boost_USE_MULTITHREADED ON)
if (BOOST_CUSTOM)
    set (Boost_FOUND true)
else ()
    set (Boost_COMPONENTS filesystem regex system thread)
    if (USE_BOOST_WAVE)
        list (APPEND Boost_COMPONENTS wave)
    endif ()

    find_package (Boost 1.34 REQUIRED 
                  COMPONENTS ${Boost_COMPONENTS}
                 )
endif ()

message (STATUS "Boost found ${Boost_FOUND} ")
message (STATUS "Boost include dirs ${Boost_INCLUDE_DIRS}")
message (STATUS "Boost library dirs ${Boost_LIBRARY_DIRS}")
message (STATUS "Boost libraries    ${Boost_LIBRARIES}")

include_directories ("${Boost_INCLUDE_DIRS}")
link_directories ("${Boost_LIBRARY_DIRS}")

# end Boost setup
###########################################################################

###########################################################################
# OpenGL setup

if (USE_OPENGL)
    find_package (OpenGL)
endif ()
message (STATUS "OPENGL_FOUND=${OPENGL_FOUND} USE_OPENGL=${USE_OPENGL}")

# end OpenGL setup
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
# GL Extension Wrangler library setup

if (USE_OPENGL)
    set (GLEW_VERSION 1.5.1)
    find_library (GLEW_LIBRARIES
                  NAMES GLEW)
    find_path (GLEW_INCLUDES
               NAMES glew.h
               PATH_SUFFIXES GL)
    if (GLEW_INCLUDES AND GLEW_LIBRARIES)
        set (GLEW_FOUND TRUE)
        message (STATUS "GLEW includes = ${GLEW_INCLUDES}")
        message (STATUS "GLEW library = ${GLEW_LIBRARIES}")
    else ()
        message (STATUS "GLEW not found")
    endif ()
endif (USE_OPENGL)

# end GL Extension Wrangler library setup
###########################################################################


###########################################################################
# LLVM library setup
if (LLVM_CUSTOM)
  message (STATUS "LLVM_DIRECTORY is ${LLVM_DIRECTORY}")
  find_library ( LLVM_LIBRARY
    NAMES LLVM-${LLVM_VERSION} LLVM-${LLVM_VERSION}svn
    PATHS ${LLVM_DIRECTORY}/lib )
  find_path ( LLVM_INCLUDES llvm/LLVMContext.h
    PATHS ${LLVM_DIRECTORY}/include
    )
else ()
  find_library ( LLVM_LIBRARY
    NAMES LLVM-2.7 LLVM-2.8svn LLVM-2.8 LLVM-2.9svn
    PATHS /usr/local/lib /opt/local/lib )
  find_path ( LLVM_INCLUDES llvm/LLVMContext.h
    PATHS /usr/local/include /opt/local/include
    )
endif ()

if (LLVM_LIBRARY AND LLVM_INCLUDES)
  set (LLVM_FOUND TRUE)
  message (STATUS "LLVM includes = ${LLVM_INCLUDES}")
  message (STATUS "LLVM library = ${LLVM_LIBRARY}")
  # ensure include directory is added (in case of non-standard locations
  if (NOT LLVM_DIRECTORY)
    # Set it to ${LLVM_LIBRARY}/../ (above lib/ and bin/)
    GET_FILENAME_COMPONENT (LLVM_LIB_DIR "${LLVM_LIBRARY}" PATH)
    GET_FILENAME_COMPONENT (LLVM_DIRECTORY "${LLVM_LIB_DIR}/../" ABSOLUTE)
  endif ()

  include_directories ("${LLVM_INCLUDES}")
else ()
  message(FATAL_ERROR "LLVM not found.")
endif ()

# end LLVM library setup
###########################################################################
