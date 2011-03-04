project (OSL)

# Release version of the library
set (OSL_LIBRARY_VERSION_MAJOR 0)
set (OSL_LIBRARY_VERSION_MINOR 5)
set (OSL_LIBRARY_VERSION_PATCH 0)

# Version of the OSO file format and instruction set
set (OSO_FILE_VERSION_MAJOR 1)
set (OSO_FILE_VERSION_MINOR 0)


cmake_minimum_required (VERSION 2.6)
set (CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)
message (STATUS "Project source dir = ${PROJECT_SOURCE_DIR}")
message (STATUS "Project build dir = ${CMAKE_BINARY_DIR}")

if (${PROJECT_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message (FATAL_ERROR "Not allowed to run in-source build!")
endif ()


if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release")
endif ()

## turn on more detailed warnings
set ( CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror" )

## disable RTTI
## NOTE(ckulla): LLVM builds without RTTI by default so beware
##   if you find the need to turn this on, as OSL classes
##   extending LLVM ones will cause linker errors (unless you
##   rebuild LLVM with rtti enabled)
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti" )
# NOTE(boulos): Boost can only detect that you've disabled RTTI for
# gcc >= 4.3. My poor mac doesn't have that.
add_definitions("-DBOOST_NO_RTTI -DBOOST_NO_TYPEID")

set (USE_TBB ON CACHE BOOL "Use TBB if needed")

set (CMAKE_MODULE_PATH
     "${PROJECT_SOURCE_DIR}/cmake/modules"
     "${PROJECT_SOURCE_DIR}/cmake")

include (util_macros)
include (platform)
include (oiio)
include (externalpackages)
include (flexbison)
include_directories (
      ${CMAKE_SOURCE_DIR}/include
      ${CMAKE_BINARY_DIR}/include
      ${OPENIMAGEIO_INCLUDES} ${TBB_INCLUDES}
  )


# We want CTest for testing
include (CTest)

include_directories ("include")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions ("-DDEBUG=1")
endif ()

if (OSL_NAMESPACE)
    add_definitions ("-DOSL_NAMESPACE=${OSL_NAMESPACE}")
endif ()

# use, i.e. don't skip the full RPATH for the build tree
set (CMAKE_SKIP_BUILD_RPATH  FALSE)
# when building, don't use the install RPATH already
# (but later on when installing)
set (CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
# the RPATH to be used when installing
set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

message (STATUS "CMAKE_INSTALL_RPATH = ${CMAKE_INSTALL_RPATH}")


###########################################################################
if (MSVC)
    add_definitions (-D_CRT_SECURE_NO_DEPRECATE)
    add_definitions (-D_CRT_SECURE_NO_WARNINGS)
    add_definitions (-D_CRT_NONSTDC_NO_WARNINGS)
    add_definitions (-D_SCL_SECURE_NO_WARNINGS)
endif (MSVC)


# Tell CMake to process the sub-directories
add_subdirectory (liboslcomp)
add_subdirectory (liboslquery)
add_subdirectory (liboslexec)

add_subdirectory (oslc)
add_subdirectory (shaders)
add_subdirectory (oslinfo)
add_subdirectory (testshade)

add_subdirectory (include)
add_subdirectory (doc)



#########################################################################
# Testing

macro ( TESTSUITE )
    foreach (_testname ${ARGN})
        add_test ( ${_testname} env OPENIMAGEIOHOME=${OPENIMAGEIOHOME} python ${PROJECT_SOURCE_DIR}/../testsuite/${_testname}/run.py ${PROJECT_SOURCE_DIR}/../testsuite/${_testname} ${CMAKE_BINARY_DIR} )
    endforeach ()
endmacro ()

# List all the individual testsuite tests here, except those that need
# special installed tests.
#TESTSUITE ( oslc-empty )
TESTSUITE ( arithmetic array array-derivs
            blendmath cellnoise closure color comparison
            derivs error-dupes exponential
            function-simple function-outputelem
            geomath gettextureinfo hyperb
            ieee_fp if incdec initops intbits layers layers-lazy
            logic loop matrix message miscmath missing-shader noise pnoise
            oslc-err-paramdefault raytype shortcircuit spline string 
            struct struct-err struct-layers struct-with-array ternary
            texture-alpha texture-blur texture-field3d
            texture-firstchannel texture-simple
            texture-width texture-withderivs texture-wrap
            transform transformc trig typecast vecctr vector xml )



#########################################################################
# Packaging
set (CPACK_PACKAGE_VERSION_MAJOR ${OSL_LIBRARY_VERSION_MAJOR})
set (CPACK_PACKAGE_VERSION_MINOR ${OSL_LIBRARY_VERSION_MINOR})
set (CPACK_PACKAGE_VERSION_PATCH ${OSL_LIBRARY_VERSION_PATCH})
# "Vendor" is only used in copyright notices, so we use the same thing that
# the rest of the copyright notices say.
set (CPACK_PACKAGE_VENDOR "Sony Pictures Imageworks")
set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "OpenShadingLanguage is...")
set (CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/doc/Description.txt")
set (CPACK_PACKAGE_FILE_NAME OSL-${OSL_LIBRARY_VERSION_MAJOR}.${OSL_LIBRARY_VERSION_MINOR}.${OSL_LIBRARY_VERSION_PATCH}-${platform})
#set (CPACK_PACKAGE_INSTALL_DIRECTORY ${PROJECT_SOURCE_DIR}/..)
exec_program ("cmake -E copy ${PROJECT_SOURCE_DIR}/../LICENSE ${CMAKE_BINARY_DIR}/License.txt")
set (CPACK_RESOURCE_FILE_LICENSE "${CMAKE_BINARY_DIR}/License.txt")
exec_program ("cmake -E copy ${PROJECT_SOURCE_DIR}/../README ${CMAKE_BINARY_DIR}/Readme.txt")
set (CPACK_RESOURCE_FILE_README "${CMAKE_BINARY_DIR}/Readme.txt")
set (CPACK_RESOURCE_FILE_WELCOME "${PROJECT_SOURCE_DIR}/doc/Welcome.txt")
#set (CPACK_PACKAGE_EXECUTABLES I'm not sure what this is for)
#set (CPACK_STRIP_FILES Do we need this?)
if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set (CPACK_GENERATOR "TGZ;STGZ;RPM;DEB")
    set (CPACK_SOURCE_GENERATOR "TGZ")
endif ()
if (APPLE)
    set (CPACK_GENERATOR "TGZ;STGZ;PackageMaker")
    set (CPACK_SOURCE_GENERATOR "TGZ")
endif ()
set (CPACK_SOURCE_PACKAGE_FILE_NAME OSL-${OSL_LIBRARY_VERSION_MAJOR}.${OSL_LIBRARY_VERSION_MINOR}.${OSL_LIBRARY_VERSION_PATCH}-source)
#set (CPACK_SOURCE_STRIP_FILES ...FIXME...)
set (CPACK_SOURCE_IGNORE_FILES ".*~")
include (CPack)
