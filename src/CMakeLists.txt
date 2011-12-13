project (OSL)

# Release version of the library
set (OSL_LIBRARY_VERSION_MAJOR 1)
set (OSL_LIBRARY_VERSION_MINOR 0)
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

# Figure out which compiler we're using
if (CMAKE_COMPILER_IS_GNUCC)
    execute_process (COMMAND ${CMAKE_C_COMPILER} -dumpversion
                     OUTPUT_VARIABLE GCC_VERSION
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    message (STATUS "Using gcc ${GCC_VERSION} as the compiler")
endif ()
if (NOT CMAKE_COMPILER_IS_CLANG)
    string (REGEX MATCH clang CMAKE_COMPILER_IS_CLANG ${CMAKE_CXX_COMPILER})
    if (CMAKE_COMPILER_IS_CLANG)
        set (CMAKE_COMPILER_IS_CLANG 1)
        message (STATUS "Using clang as the compiler")
    endif ()
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

# For gcc < 4.3, Boost needs some extra help detecting that we've
# disabled RTTI.
if (CMAKE_COMPILER_IS_GNUCC AND GCC_VERSION VERSION_LESS 4.3)
    add_definitions ("-DBOOST_NO_RTTI")
    add_definitions ("-DBOOST_NO_TYPEID")
endif ()

# Turn off a few pesky clang warnings that we shouldn't allow to be errors.
# Mostly these are warnings it generates in boost headers.  I'd love to
# turn off the warnings just for external code, not project code, but I
# don't think there's a way to do this.
if (CMAKE_COMPILER_IS_CLANG)
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-parentheses-equality")
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-char-subscripts")
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
endif ()

set (USE_TBB ON CACHE BOOL "Use TBB if needed")
if (WIN32)
    set (USE_BOOST_WAVE ON CACHE BOOL "Use Boost Wave as preprocessor")
else ()
    set (USE_BOOST_WAVE OFF CACHE BOOL "Use Boost Wave as preprocessor")
endif ()

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
        # Run the test unoptimized
        add_test ( ${_testname} env OPENIMAGEIOHOME=${OPENIMAGEIOHOME} TESTSHADE_OPT=0 python ${PROJECT_SOURCE_DIR}/../testsuite/${_testname}/run.py ${PROJECT_SOURCE_DIR}/../testsuite/${_testname} ${CMAKE_BINARY_DIR} )
        # Run the same test again with aggressive -O2 runtime
        # optimization, triggered by setting TESTSHADE_OPT env variable
        add_test ( ${_testname}.opt env OPENIMAGEIOHOME=${OPENIMAGEIOHOME} TESTSHADE_OPT=2 python ${PROJECT_SOURCE_DIR}/../testsuite/${_testname}/run.py ${PROJECT_SOURCE_DIR}/../testsuite/${_testname} ${CMAKE_BINARY_DIR} )
    endforeach ()
endmacro ()

# List all the individual testsuite tests here, except those that need
# special installed tests.
#TESTSUITE ( oslc-empty )
TESTSUITE ( arithmetic array array-derivs array-range
            blackbody blendmath breakcont bug-locallifetime
            cellnoise closure color comparison
            component-range const-array-params debugnan
            derivs derivs-muldiv-clobber error-dupes exponential
            function-earlyreturn function-simple function-outputelem
            geomath getsymbol-nonheap gettextureinfo hyperb
            ieee_fp if incdec initops intbits layers layers-lazy
            logic loop matrix message miscmath missing-shader
            noise noise-cell
            noise-gabor noise-gabor2d-filter noise-gabor3d-filter
            noise-perlin noise-uperlin
            pnoise pnoise-cell pnoise-perlin pnoise-uperlin
            oslc-err-noreturn oslc-err-paramdefault
            raytype shortcircuit spline splineinverse string 
            struct struct-array struct-array-mixture
            struct-err struct-layers struct-with-array 
            struct-within-struct ternary
            texture-alpha texture-blur texture-field3d
            texture-firstchannel texture-interp texture-simple
            texture-width texture-withderivs texture-wrap
            transform transformc trig typecast vecctr vector
            wavelength_color xml )



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
