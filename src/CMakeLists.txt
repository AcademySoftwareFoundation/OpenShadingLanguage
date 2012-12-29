project (OSL)

# Release version of the library
set (OSL_LIBRARY_VERSION_MAJOR 1)
set (OSL_LIBRARY_VERSION_MINOR 3)
set (OSL_LIBRARY_VERSION_PATCH 0)

# Version of the OSO file format and instruction set
set (OSO_FILE_VERSION_MAJOR 1)
set (OSO_FILE_VERSION_MINOR 0)

cmake_minimum_required (VERSION 2.6)
if (NOT CMAKE_VERSION VERSION_LESS 2.8.4)
    cmake_policy (SET CMP0017 NEW)
endif ()
set (CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)
message (STATUS "Project source dir = ${PROJECT_SOURCE_DIR}")
message (STATUS "Project build dir = ${CMAKE_BINARY_DIR}")

if (${PROJECT_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message (FATAL_ERROR "Not allowed to run in-source build!")
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release")
endif ()
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions ("-DDEBUG=1")
    set (DEBUGMODE ON)
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

## turn on more detailed warnings and consider warnings as errors
set (STOP_ON_WARNING ON CACHE BOOL "Stop building if there are any compiler warnings")
if (NOT MSVC)
    add_definitions ("-Wall")
    if (STOP_ON_WARNING)
        add_definitions ("-Werror")
    endif ()
endif ()

## disable RTTI
## NOTE(ckulla): LLVM builds without RTTI by default so beware
##   if you find the need to turn this on, as OSL classes
##   extending LLVM ones will cause linker errors (unless you
##   rebuild LLVM with rtti enabled)
if (MSVC)
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR-" )
else ()
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti" )
endif()

# For gcc < 4.3 and certain Clang versions, Boost needs some extra help
# detecting that we've disabled RTTI.
if ((CMAKE_COMPILER_IS_GNUCC AND GCC_VERSION VERSION_LESS 4.3)
    OR CMAKE_COMPILER_IS_CLANG)
    add_definitions ("-DBOOST_NO_RTTI")
    add_definitions ("-DBOOST_NO_TYPEID")
endif ()

# Needed for static boost libraries on Windows
if (WIN32 AND Boost_USE_STATIC_LIBS)
    add_definitions ("-DBOOST_ALL_NO_LIB")
    add_definitions ("-DBOOST_THREAD_USE_LIB")
endif ()

# Needed to disable dllimport/dllexport for static library
if (BUILDSTATIC)
    add_definitions ("-DOSL_STATIC_LIBRARY")
endif()

if (CMAKE_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_GNUCC)
    # CMake doesn't automatically know what do do with
    # include_directories(SYSTEM...) when using clang or gcc.
    set (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
endif ()

if (CMAKE_COMPILER_IS_CLANG)
    # disable warning about unused command line arguments
    add_definitions ("-Qunused-arguments -Wno-unknown-pragmas")
    # disable warning in flex-generated code
    add_definitions ("-Wno-null-conversion")
endif ()

if (CMAKE_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_GNUCC)
    if (HIDE_SYMBOLS AND NOT DEBUGMODE)
        # Turn default symbol visibility to hidden
        set (VISIBILITY_COMMAND "-fvisibility=hidden")
        add_definitions (${VISIBILITY_COMMAND})
    endif ()
    if (LLVM_STATIC AND ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND NOT DEBUGMODE)
        # Linux: When linking against LLVM statically, we can also hide
        # all its symbols to prevent clashes if OSL is linked against an
        # app that also embeds LLVM.
        set (VISIBILITY_MAP_COMMAND "-Wl,--version-script=${PROJECT_SOURCE_DIR}/liboslexec/liboslexec.map")
    endif ()
endif ()


set (VERBOSE OFF CACHE BOOL "Print lots of messages while compiling")
set (BUILDSTATIC OFF CACHE BOOL "Build static library instead of shared")
set (HIDE_SYMBOLS OFF CACHE BOOL "Hide symbols not in the public API")
set (USE_TBB ON CACHE BOOL "Use TBB if needed")
if (WIN32)
    set (USE_BOOST_WAVE ON CACHE BOOL "Use Boost Wave as preprocessor")
else ()
    set (USE_BOOST_WAVE OFF CACHE BOOL "Use Boost Wave as preprocessor")
endif ()
set (USE_PARTIO ON CACHE BOOL "Use Partio if found")

if (LLVM_NAMESPACE)
    add_definitions ("-DLLVM_NAMESPACE=${LLVM_NAMESPACE}")
endif ()

if (EXTRA_CPP_DEFINITIONS)
    add_definitions ("${EXTRA_CPP_DEFINITIONS}")
endif()


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


include_directories ("include")

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


# We want CTest for testing
# N.B. This needs to be added before any of the subdirectories, or
# their add_test commands will not register.
include (CTest)


# Tell CMake to process the sub-directories
add_subdirectory (liboslcomp)
add_subdirectory (liboslquery)
add_subdirectory (liboslexec)

add_subdirectory (oslc)
add_subdirectory (shaders)
add_subdirectory (oslinfo)
add_subdirectory (testshade)
add_subdirectory (testrender)

add_subdirectory (include)
add_subdirectory (doc)



#########################################################################
# Testing

# Make a copy of the testsuite into the build area
if (DEFINED CMAKE_VERSION AND NOT CMAKE_VERSION VERSION_LESS 2.8)
    file (COPY ${PROJECT_SOURCE_DIR}/../testsuite
          DESTINATION ${CMAKE_BINARY_DIR})
endif()

macro ( TESTSUITE )
    parse_arguments (_ats "LABEL" "" ${ARGN})
    # If there was a FOUNDVAR param specified and that variable name is
    # not defined, mark the test as broken.
    if (_ats_FOUNDVAR AND NOT ${_ats_FOUNDVAR})
        set (_ats_LABEL "broken")
    endif ()
    # Add the tests if all is well.
    foreach (_testname ${_ats_DEFAULT_ARGS})
        set (_testdir "${CMAKE_BINARY_DIR}/testsuite/${_testname}")
        if (_ats_LABEL MATCHES "broken")
            set (_testname "${_testname}-broken")
        endif ()
        set (_runtest python ${CMAKE_BINARY_DIR}/testsuite/runtest.py
                             ${_testdir} ${_extra_test_args})
        if (MSVC_IDE)
            set (_runtest ${_runtest} --devenv-config $<CONFIGURATION>
                                      --solution-path "${CMAKE_BINARY_DIR}" )
        endif ()

        if (VERBOSE)
            message (STATUS "TEST ${_testname}: ${CMAKE_BINARY_DIR}/testsuite/runtest.py ${_testdir} ${_extra_test_args}")
        endif ()

        # Run the test unoptimized
        if (NOT _testname MATCHES "^render")
          set (_env TESTSHADE_OPT=0 OPENIMAGEIOHOME=${OPENIMAGEIOHOME})
          add_test ( NAME ${_testname}
                     COMMAND env ${_env} ${_runtest} )
        endif ()
        # Run the same test again with aggressive -O2 runtime
        # optimization, triggered by setting TESTSHADE_OPT env variable
        set (_env TESTSHADE_OPT=2 OPENIMAGEIOHOME=${OPENIMAGEIOHOME})
        add_test ( NAME ${_testname}.opt
                   COMMAND env ${_env} ${_runtest} )
    endforeach ()
endmacro ()

# List all the individual testsuite tests here, except those that need
# special installed tests.
#TESTSUITE ( oslc-empty )
TESTSUITE ( arithmetic array array-derivs array-range
            blackbody blendmath breakcont bug-locallifetime
            cellnoise closure color comparison
            component-range const-array-params debugnan
            derivs derivs-muldiv-clobber error-dupes exit exponential
            function-earlyreturn function-simple function-outputelem
            geomath getsymbol-nonheap gettextureinfo hyperb
            ieee_fp if incdec initops intbits isconnected
            layers layers-Ciassign layers-lazy
            logic loop matrix message metadata-braces miscmath missing-shader
            noise noise-cell
            noise-gabor noise-gabor2d-filter noise-gabor3d-filter
            noise-perlin noise-uperlin
            pnoise pnoise-cell pnoise-gabor pnoise-perlin pnoise-uperlin
            oslc-err-arrayindex oslc-err-closuremul oslc-err-intoverflow
            oslc-err-noreturn oslc-err-paramdefault
            pointcloud pointcloud-fold
            raytype render-background render-bumptest render-cornell render-furnace-diffuse
            render-microfacet render-oren-nayar render-veachmis render-ward
            shortcircuit spline splineinverse string 
            struct struct-array struct-array-mixture
            struct-err struct-layers struct-with-array 
            struct-within-struct ternary
            texture-alpha texture-blur texture-field3d
            texture-firstchannel texture-interp texture-simple
            texture-smallderivs texture-swirl
            texture-width texture-withderivs texture-wrap
            transform transformc trig typecast
            unknown-instruction vecctr vector
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
exec_program ("cmake -E copy ${PROJECT_SOURCE_DIR}/../README.md ${CMAKE_BINARY_DIR}/Readme.txt")
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
