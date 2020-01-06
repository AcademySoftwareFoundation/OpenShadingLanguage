cmake_minimum_required (VERSION 3.12)
project (OSL VERSION 1.11.3.0
         LANGUAGES CXX C
         HOMEPAGE_URL "https://github.com/imageworks/OpenShadingLanguage")
set (PROJ_NAME ${PROJECT_NAME})    # short name
string (TOLOWER ${PROJ_NAME} PROJ_NAME_LOWER)  # short name lower case
string (TOUPPER ${PROJ_NAME} PROJ_NAME_UPPER)  # short name upper case
set (PROJECT_VERSION_RELEASE_TYPE "dev")   # "dev", "betaX", "RCY", ""
set (${PROJECT_NAME}_VERSION_RELEASE_TYPE ${PROJECT_VERSION_RELEASE_TYPE})
set (PROJECT_AUTHORS "Sony Pictures Imageworks, et al")

message (STATUS "Building ${PROJECT_NAME} ${PROJECT_VERSION}")
message (STATUS "CMake version is ${CMAKE_VERSION}")

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release")
endif ()

message (STATUS "Configuring ${PROJECT_NAME} ${PROJECT_VERSION}")
message (STATUS "CMake ${CMAKE_VERSION}")
message (STATUS "CMake system           = ${CMAKE_SYSTEM}")
message (STATUS "CMake system name      = ${CMAKE_SYSTEM_NAME}")
message (STATUS "Project source dir     = ${PROJECT_SOURCE_DIR}")
message (STATUS "Project build dir      = ${CMAKE_BINARY_DIR}")
message (STATUS "Project install prefix = ${CMAKE_INSTALL_PREFIX}")
message (STATUS "Configuration types    = ${CMAKE_CONFIGURATION_TYPES}")
message (STATUS "Build type             = ${CMAKE_BUILD_TYPE}")

if ("${PROJECT_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message (FATAL_ERROR "Not allowed to run in-source build!")
endif ()


option (CMAKE_USE_FOLDERS "Use the FOLDER target property to organize targets into folders." ON)
mark_as_advanced (CMAKE_USE_FOLDERS)
if (CMAKE_USE_FOLDERS)
    set_property (GLOBAL PROPERTY USE_FOLDERS ON)
endif ()


# Version of the OSO file format and instruction set
set (OSO_FILE_VERSION_MAJOR 1)
set (OSO_FILE_VERSION_MINOR 0)


# This needs to be early, for CMAKE_INSTALL_FULL_DATADIR
include (GNUInstallDirs)


option (VERBOSE "Print lots of messages while compiling" OFF)
set (${PROJ_NAME}_NAMESPACE ${PROJECT_NAME} CACHE STRING
     "Customized outer namespace base name (version will be added)")
option (${PROJ_NAME}_NAMESPACE_INCLUDE_PATCH "Should the inner namespace include the patch number" OFF)
set (SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
     CACHE STRING "Set the SO version in the SO name of the output library")
set (OSL_LIBNAME_SUFFIX "" CACHE STRING
     "Optional name appended to ${PROJECT_NAME} libraries that are built")
option (OSL_BUILD_TESTS "Build the unit tests, testshade, testrender" ON)
if (WIN32)
    option (USE_LLVM_BITCODE "Generate embedded LLVM bitcode" OFF)
else ()
    option (USE_LLVM_BITCODE "Generate embedded LLVM bitcode" ON)
endif ()
option (OSL_BUILD_PLUGINS "Bool OSL plugins, for example OIIO plugin" ON)
option (OSL_BUILD_SHADERS "Build shaders" ON)
option (OSL_BUILD_MATERIALX "Build MaterialX shaders" ON)
option (USE_OPTIX "Include OptiX support" OFF)
set (OPTIX_EXTRA_LIBS CACHE STRING "Extra lib targets needed for OptiX")
set (CUDA_TARGET_ARCH "sm_35" CACHE STRING "CUDA GPU architecture (e.g. sm_35)")
set (OSL_SHADER_INSTALL_DIR "${CMAKE_INSTALL_FULL_DATADIR}/${PROJECT_NAME}/shaders"
     CACHE STRING "Directory where shaders will be installed")
set (OSL_PTX_INSTALL_DIR "${CMAKE_INSTALL_FULL_DATADIR}/${PROJECT_NAME}/ptx"
     CACHE STRING "Directory where OptiX PTX files will be installed")
set (CMAKE_DEBUG_POSTFIX "" CACHE STRING "Library naming postfix for Debug builds (e.g., '_debug')")


set (OSL_NO_DEFAULT_TEXTURESYSTEM OFF CACHE BOOL "Do not use create a raw OIIO::TextureSystem")
if (OSL_NO_DEFAULT_TEXTURESYSTEM)
    add_definitions ("-DOSL_NO_DEFAULT_TEXTURESYSTEM=1")
endif ()

option (USE_FAST_MATH "Use fast math approximations (if no, then use system math library)" ON)
if (USE_FAST_MATH)
    add_definitions ("-DOSL_FAST_MATH=1")
else ()
    add_definitions ("-DOSL_FAST_MATH=0")
endif ()

if (USE_OPTIX)
    add_definitions ("-DOSL_USE_OPTIX=1")
endif ()


# Set the default namespace. For symbol hiding reasons, it's important that
# the project name is a subset of the namespace name.
set (PROJ_NAMESPACE "${${PROJ_NAME}_NAMESPACE}")
string(REGEX MATCH ${PROJECT_NAME} NAMESPACE_HAS_PROJECT_NAME ${PROJ_NAMESPACE})
if (NOT NAMESPACE_HAS_PROJECT_NAME)
    set (PROJ_NAMESPACE ${PROJECT_NAME}_${PROJ_NAMESPACE})
endif ()
set (PROJ_NAMESPACE_V "${PROJ_NAMESPACE}_v${PROJECT_VERSION_MAJOR}_${PROJECT_VERSION_MINOR}")
if (OIIO_NAMESPACE_INCLUDE_PATCH)
    set (PROJ_NAMESPACE_V "${PROJ_NAMESPACE_V}_${PROJECT_VERSION_PATCH}")
endif ()
message(STATUS "Setting Namespace to: ${PROJ_NAMESPACE_V}")


list (APPEND CMAKE_MODULE_PATH
      "${PROJECT_SOURCE_DIR}/src/cmake/modules"
      "${PROJECT_SOURCE_DIR}/src/cmake")

# Helpful functions and macros for our project
include (colors)
include (check_is_enabled)

# All the C++ and compiler related options and adjustments live here
include (compiler)

include (externalpackages)
include (flexbison)
include (cuda_macros)

# We want CTest for testing, so this must be added before all the calls to
# add_subdirectory, or their add_test commands will not register.
include (CTest)


include_directories (
    BEFORE
    "${CMAKE_SOURCE_DIR}/src/include"
    "${CMAKE_BINARY_DIR}/src/include"
    "${CMAKE_BINARY_DIR}/include"
  )


# Tell CMake to process the sub-directories
add_subdirectory (src/include)
add_subdirectory (src/liboslcomp)
add_subdirectory (src/liboslquery)
add_subdirectory (src/liboslexec)
add_subdirectory (src/liboslnoise)
add_subdirectory (src/oslc)
add_subdirectory (src/oslinfo)

if (OSL_BUILD_TESTS)
    add_subdirectory (src/testshade)
    add_subdirectory (src/testrender)
endif ()

if (OSL_BUILD_PLUGINS)
    add_subdirectory (src/osl.imageio)
endif ()

if (USE_QT AND Qt5_FOUND)
    add_subdirectory (src/osltoy)
endif ()

if (OSL_BUILD_SHADERS)
    add_subdirectory (src/shaders)
    if (OSL_BUILD_MATERIALX)
      add_subdirectory (src/shaders/MaterialX)
    endif ()
endif ()

option (INSTALL_DOCS "Install documentation" ON)
if (INSTALL_DOCS)
    add_subdirectory (src/doc)
endif ()

# Last minute site-specific instructions, if they exist
if (OSL_SITE AND EXISTS "${PROJECT_SOURCE_DIR}/site/${OSL_SITE}/cmake/sitecustom.cmake")
    include ("${PROJECT_SOURCE_DIR}/site/${OSL_SITE}/cmake/sitecustom.cmake")
endif ()

# install pkgconfig files
if ( NOT MSVC )
   configure_file(src/build-scripts/oslexec.pc.in "${CMAKE_BINARY_DIR}/oslexec.pc" @ONLY)
   configure_file(src/build-scripts/oslcomp.pc.in "${CMAKE_BINARY_DIR}/oslcomp.pc" @ONLY)
   configure_file(src/build-scripts/oslquery.pc.in "${CMAKE_BINARY_DIR}/oslquery.pc" @ONLY)
   install (FILES "${CMAKE_BINARY_DIR}/oslexec.pc"
                  "${CMAKE_BINARY_DIR}/oslcomp.pc"
                  "${CMAKE_BINARY_DIR}/oslquery.pc"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
            COMPONENT developer)
endif()



#########################################################################
# Export the configuration files. There are also library-specific config
# exports in the CMakeLists.txt of libOpenImageIO.
include (CMakePackageConfigHelpers)

# the file containing the exported targets
set (OSL_TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets.cmake")
# the version file
set (OSL_VERSION_CONFIG "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
# the config file that is actually looked for by find_package
set (OSL_PROJECT_CONFIG "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
# where all these files will be installed
set (OSL_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# first generate the version file in the binary dir
write_basic_package_version_file (
        ${OSL_VERSION_CONFIG}
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)

# generate the Targets file in the binary dir using the targets collected in
# OSL_EXPORTED_TARGETS each target is added to OSL_EXPORTED_TARGETS through
# the macro install_target().
export (EXPORT OSL_EXPORTED_TARGETS FILE "${CMAKE_BINARY_DIR}/${OSL_TARGETS_EXPORT_NAME}")

# generate the config file from the template in the binary dir
configure_package_config_file ("${PROJECT_SOURCE_DIR}/src/cmake/Config.cmake.in"
        "${OSL_PROJECT_CONFIG}"
        INSTALL_DESTINATION "${OSL_CONFIG_INSTALL_DIR}")

# generate the config file from the template in the binary dir
install (FILES "${OSL_PROJECT_CONFIG}" "${OSL_VERSION_CONFIG}"
        DESTINATION "${OSL_CONFIG_INSTALL_DIR}")

# install targets files
install (EXPORT OSL_EXPORTED_TARGETS
        DESTINATION ${OSL_CONFIG_INSTALL_DIR}
        FILE ${OSL_TARGETS_EXPORT_NAME}
        NAMESPACE ${PROJECT_NAME}::)





#########################################################################
# Testing

# Make a build/platform/testsuite directory, and copy the master runtest.py
# there. The rest is up to the tests themselves.
file (MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/testsuite")
add_custom_command (OUTPUT "${CMAKE_BINARY_DIR}/testsuite/runtest.py"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${CMAKE_SOURCE_DIR}/testsuite/runtest.py"
                        "${CMAKE_BINARY_DIR}/testsuite/runtest.py"
                    MAIN_DEPENDENCY "${CMAKE_SOURCE_DIR}/testsuite/runtest.py")
add_custom_target ( CopyFiles ALL DEPENDS "${CMAKE_BINARY_DIR}/testsuite/runtest.py" )

macro ( TESTSUITE )
    cmake_parse_arguments (_ats "" "LABEL" "" ${ARGN})
    # If there was a FOUNDVAR param specified and that variable name is
    # not defined, mark the test as broken.
    if (_ats_FOUNDVAR AND NOT ${_ats_FOUNDVAR})
        set (_ats_LABEL "broken")
    endif ()
    set (test_all_optix $ENV{TESTSUITE_OPTIX})
    # Add the tests if all is well.
    set (ALL_TEST_LIST "")
    foreach (_testname ${_ats_UNPARSED_ARGUMENTS})
        set (_testsrcdir "${CMAKE_SOURCE_DIR}/testsuite/${_testname}")
        set (_testdir "${CMAKE_BINARY_DIR}/testsuite/${_testname}")
        if (_ats_LABEL MATCHES "broken")
            set (_testname "${_testname}-broken")
        endif ()
        set (_runtest python "${CMAKE_BINARY_DIR}/testsuite/runtest.py"
                             ${_testdir} ${_extra_test_args})
        if (MSVC_IDE)
            set (_runtest ${_runtest} --devenv-config $<CONFIGURATION>
                                      --solution-path "${CMAKE_BINARY_DIR}" )
        endif ()

        set (ALL_TEST_LIST "${ALL_TEST_LIST} ${_testname}")
        file (MAKE_DIRECTORY "${_testdir}")

        # Run the test unoptimized, unless it matches a few patterns that
        # we don't test unoptimized (or has an OPTIMIZEONLY marker file).
        if (NOT _testname MATCHES "^getattribute-shader" AND
            NOT _testname MATCHES "optix" AND
            NOT EXISTS "${_testsrcdir}/OPTIMIZEONLY")
            set (_env TESTSHADE_OPT=0
                      OpenImageIO_ROOT=${OpenImageIO_ROOT}
                      OSL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                      OSL_BUILD_DIR=${CMAKE_BINARY_DIR} )
            add_test ( NAME ${_testname}
                       COMMAND env ${_env} ${_runtest} )
        endif ()
        # Run the same test again with aggressive -O2 runtime
        # optimization, triggered by setting TESTSHADE_OPT env variable.
        # Skip OptiX-only tests and those with a NOOPTIMIZE marker file.
        if (NOT _testname MATCHES "optix"
            AND NOT EXISTS "${_testsrcdir}/NOOPTIMIZE")
            set (_env TESTSHADE_OPT=2
                      OpenImageIO_ROOT=${OpenImageIO_ROOT}
                      OSL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                      OSL_BUILD_DIR=${CMAKE_BINARY_DIR} )
            add_test ( NAME ${_testname}.opt
                       COMMAND env ${_env} ${_runtest} )
        endif ()
        # When building for OptiX support, also run it in OptiX mode
        # if there is an OPTIX marker file in the directory.
        # If an environment variable $TESTSUITE_OPTIX is nonzero, then
        # run all tests with OptiX, even if there's no OPTIX marker.
        if (USE_OPTIX
            AND (EXISTS "${_testsrcdir}/OPTIX" OR test_all_optix OR _testname MATCHES "optix")
            AND NOT EXISTS "${_testsrcdir}/NOOPTIX"
            AND NOT EXISTS "${_testsrcdir}/NOOPTIX-FIXME")
            # Unoptimized
            if (NOT EXISTS "${_testsrcdir}/OPTIMIZEONLY")
                set (_env TESTSHADE_OPT=0 TESTSHADE_OPTIX=1
                          OpenImageIO_ROOT=${OpenImageIO_ROOT}
                          OSL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                          OSL_BUILD_DIR=${CMAKE_BINARY_DIR} )
                add_test ( NAME ${_testname}.optix
                           COMMAND env ${_env} ${_runtest} )
            endif ()
            # and optimized
            set (_env TESTSHADE_OPT=2 TESTSHADE_OPTIX=1
                      OpenImageIO_ROOT=${OpenImageIO_ROOT}
                      OSL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                      OSL_BUILD_DIR=${CMAKE_BINARY_DIR} )
            add_test ( NAME ${_testname}.optix.opt
                       COMMAND env ${_env} ${_runtest} )
        endif ()
    endforeach ()
    if (VERBOSE)
        message (STATUS "Added tests: ${ALL_TEST_LIST}")
    endif ()
endmacro ()

if (OSL_BUILD_TESTS)
# List all the individual testsuite tests here, except those that need
# special installed tests.
TESTSUITE ( aastep allowconnect-err and-or-not-synonyms arithmetic
            array array-derivs array-range array-aassign
            blackbody blendmath breakcont
            bug-array-heapoffsets bug-locallifetime bug-outputinit
            bug-param-duplicate bug-peep bug-return
            cellnoise closure closure-array color comparison
            compile-buffer
            component-range
            connect-components
            const-array-params const-array-fill
            debugnan debug-uninit
            derivs derivs-muldiv-clobber
            draw_string
            error-dupes error-serialized
            exit exponential
            fprintf
            function-earlyreturn function-simple function-outputelem
            function-overloads function-redef
            geomath getattribute-camera getattribute-shader
            getsymbol-nonheap gettextureinfo
            group-outputs groupstring
            hash hashnoise hex hyperb
            ieee_fp if incdec initlist initops intbits isconnected isconstant
            layers layers-Ciassign layers-entry layers-lazy
            layers-nonlazycopy layers-repeatedoutputs
            linearstep
            logic loop matrix message
            mergeinstances-nouserdata mergeinstances-vararray
            metadata-braces miscmath missing-shader
            named-components
            noise noise-cell
            noise-gabor noise-gabor2d-filter noise-gabor3d-filter
            noise-perlin noise-simplex
            pnoise pnoise-cell pnoise-gabor pnoise-perlin
            operator-overloading
            opt-warnings
            oslc-comma oslc-D oslc-M
            oslc-err-arrayindex oslc-err-assignmenttypes
            oslc-err-closuremul oslc-err-field
            oslc-err-format oslc-err-funcoverload
            oslc-err-intoverflow oslc-err-write-nonoutput
            oslc-err-noreturn oslc-err-notfunc
            oslc-err-initlist-args oslc-err-initlist-return
            oslc-err-named-components
            oslc-err-outputparamvararray oslc-err-paramdefault
            oslc-err-struct-array-init oslc-err-struct-ctr
            oslc-err-struct-dup oslc-err-struct-print
            oslc-err-unknown-ctr
            oslc-warn-commainit
            oslc-variadic-macro
            oslc-version
            oslinfo-arrayparams oslinfo-colorctrfloat
            oslinfo-metadata oslinfo-noparams
            osl-imageio
            paramval-floatpromotion
            pragma-nowarn
            printf-whole-array
            raytype raytype-specialized reparam
            render-background render-bumptest
            render-cornell render-furnace-diffuse
            render-microfacet render-oren-nayar render-veachmis render-ward
            select shortcircuit spline splineinverse splineinverse-ident
            spline-boundarybug spline-derivbug
            string
            struct struct-array struct-array-mixture
            struct-err struct-init-copy
            struct-isomorphic-overload struct-layers
            struct-operator-overload struct-return struct-with-array
            struct-nested struct-nested-assign struct-nested-deep
            ternary
            testshade-expr
            texture-alpha texture-blur texture-connected-options
            texture-derivs texture-errormsg
            texture-firstchannel texture-interp
            texture-missingalpha texture-missingcolor texture-simple
            texture-smallderivs texture-swirl texture-udim
            texture-width texture-withderivs texture-wrap
            trailing-commas
            transitive-assign
            transform transformc trig typecast
            unknown-instruction
            vararray-connect vararray-default
            vararray-deserialize vararray-param
            vecctr vector
            wavelength_color Werror xml )

# Only run field3d-related tests if the local OIIO was built with f3d support.
EXECUTE_PROCESS ( COMMAND ${OPENIMAGEIO_BIN} --help
                  OUTPUT_VARIABLE oiiotool_help )
if (oiiotool_help MATCHES "field3d")
    TESTSUITE ( texture-field3d )
endif()

# Only run pointcloud tests if Partio is found
if (PARTIO_FOUND)
    TESTSUITE ( pointcloud pointcloud-fold )
endif ()

# Only run the OptiX tests if OptiX and CUDA are found
if (OPTIX_FOUND AND CUDA_FOUND)
    TESTSUITE ( testoptix testoptix-noise )
endif ()

# FIXME: still working on MaterialX testsuite
# add_subdirectory(testsuite/MaterialX)

endif (OSL_BUILD_TESTS)



#########################################################################
# Packaging
set (CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set (CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set (CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
# "Vendor" is only used in copyright notices, so we use the same thing that
# the rest of the copyright notices say.
set (CPACK_PACKAGE_VENDOR ${PROJECT_AUTHORS})
set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "OpenShadingLanguage is...")
set (CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/src/doc/Description.txt")
set (CPACK_PACKAGE_FILE_NAME ${PROJECT_NAME}-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}-${platform})
file (COPY "${PROJECT_SOURCE_DIR}/LICENSE" DESTINATION "${CMAKE_BINARY_DIR}")
file (RENAME "${CMAKE_BINARY_DIR}/LICENSE" "${CMAKE_BINARY_DIR}/License.txt")
set (CPACK_RESOURCE_FILE_LICENSE "${CMAKE_BINARY_DIR}/License.txt")
file (COPY "${PROJECT_SOURCE_DIR}/README.md" DESTINATION "${CMAKE_BINARY_DIR}")
file (RENAME "${CMAKE_BINARY_DIR}/README.md" "${CMAKE_BINARY_DIR}/Readme.txt")
set (CPACK_RESOURCE_FILE_README "${CMAKE_BINARY_DIR}/Readme.txt")
set (CPACK_RESOURCE_FILE_WELCOME "${PROJECT_SOURCE_DIR}/src/doc/Welcome.txt")
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
set (CPACK_SOURCE_PACKAGE_FILE_NAME ${PROJECT_NAME}-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}-source)
#set (CPACK_SOURCE_STRIP_FILES ...FIXME...)
set (CPACK_SOURCE_IGNORE_FILES ".*~")
include (CPack)
