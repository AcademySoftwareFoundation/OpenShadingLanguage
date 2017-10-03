project (OSL)

# Release version of the library
set (PROJ_NAME OSL)
set (PROJECT_VERSION_MAJOR 1)
set (PROJECT_VERSION_MINOR 8)
set (PROJECT_VERSION_PATCH 12)
set (OSL_LIBRARY_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set (OSL_LIBRARY_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set (OSL_LIBRARY_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set (OSL_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set (OSL_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set (OSL_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set (OSL_LIBRARY_VERSION_RELEASE_TYPE "")   # "dev", "betaX", "RCY", ""

# Version of the OSO file format and instruction set
set (OSO_FILE_VERSION_MAJOR 1)
set (OSO_FILE_VERSION_MINOR 0)

message (STATUS "CMake version is ${CMAKE_VERSION}")
cmake_minimum_required (VERSION 3.0)
cmake_policy (SET CMP0017 NEW)
cmake_policy (SET CMP0025 NEW)  # Detect AppleClang for new CMake
if (NOT CMAKE_VERSION VERSION_LESS 3.2.2)
    # cmake_policy (SET CMP0042 OLD)
    cmake_policy (SET CMP0046 OLD)
endif ()
set (CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)

if (VERBOSE)
    message (STATUS "Project source dir = ${PROJECT_SOURCE_DIR}")
endif ()
message (STATUS "Project build dir = ${CMAKE_BINARY_DIR}")

if ("${PROJECT_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message (FATAL_ERROR "Not allowed to run in-source build!")
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release")
endif ()
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set (DEBUGMODE ON)
endif ()
if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    # cmake bug workaround -- on some platforms, cmake doesn't set
    # NDEBUG for RelWithDebInfo mode
    add_definitions ("-DNDEBUG")
endif ()

option (CMAKE_USE_FOLDERS "Use the FOLDER target property to organize targets into folders." ON)
mark_as_advanced (CMAKE_USE_FOLDERS)
if (CMAKE_USE_FOLDERS)
    set_property (GLOBAL PROPERTY USE_FOLDERS ON)
endif ()

## turn on more detailed warnings and consider warnings as errors
set (STOP_ON_WARNING ON CACHE BOOL "Stop building if there are any compiler warnings")
if (NOT MSVC)
    add_definitions ("-Wall")
    if (STOP_ON_WARNING)
        add_definitions ("-Werror")
    endif ()
endif ()

message (STATUS "CMAKE_CXX_COMPILER is ${CMAKE_CXX_COMPILER}")
message (STATUS "CMAKE_CXX_COMPILER_ID is ${CMAKE_CXX_COMPILER_ID}")

# Figure out which compiler we're using
if (CMAKE_COMPILER_IS_GNUCC)
    execute_process (COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
                     OUTPUT_VARIABLE GCC_VERSION
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (VERBOSE)
        message (STATUS "Using gcc ${GCC_VERSION} as the compiler")
    endif ()
endif ()

# Figure out which compiler we're using, for tricky cases
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER MATCHES "[Cc]lang")
    # If using any flavor of clang, set CMAKE_COMPILER_IS_CLANG. If it's
    # Apple's variety, set CMAKE_COMPILER_IS_APPLECLANG and
    # APPLECLANG_VERSION_STRING, otherwise for generic clang set
    # CLANG_VERSION_STRING.
    set (CMAKE_COMPILER_IS_CLANG 1)
    EXECUTE_PROCESS( COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE clang_full_version_string )
    if (clang_full_version_string MATCHES "Apple")
        set (CMAKE_CXX_COMPILER_ID "AppleClang")
        set (CMAKE_COMPILER_IS_APPLECLANG 1)
        string (REGEX REPLACE ".* version ([0-9]+\\.[0-9]+).*" "\\1" APPLECLANG_VERSION_STRING ${clang_full_version_string})
        if (VERBOSE)
            message (STATUS "The compiler is Clang: ${CMAKE_CXX_COMPILER_ID} version ${APPLECLANG_VERSION_STRING}")
        endif ()
    else ()
        string (REGEX REPLACE ".* version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION_STRING ${clang_full_version_string})
        if (VERBOSE)
            message (STATUS "The compiler is Clang: ${CMAKE_CXX_COMPILER_ID} version ${CLANG_VERSION_STRING}")
        endif ()
    endif ()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set (CMAKE_COMPILER_IS_INTEL 1)
    if (VERBOSE)
        message (STATUS "Using Intel as the compiler")
    endif ()
endif ()

# Options common to gcc and clang
if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANG)
    # CMake doesn't automatically know what do do with
    # include_directories(SYSTEM...) when using clang or gcc.
    set (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
    # Ensure this macro is set for stdint.h
    add_definitions ("-D__STDC_LIMIT_MACROS")
    # this allows native instructions to be used for sqrtf instead of a function call
    add_definitions ("-fno-math-errno")
    if (HIDE_SYMBOLS AND NOT DEBUGMODE)
        # Turn default symbol visibility to hidden
        set (VISIBILITY_COMMAND "-fvisibility=hidden -fvisibility-inlines-hidden")
        add_definitions (${VISIBILITY_COMMAND})
        if (CMAKE_SYSTEM_NAME MATCHES "Linux|kFreeBSD" OR CMAKE_SYSTEM_NAME STREQUAL "GNU")
            # Linux/FreeBSD/Hurd: also hide all the symbols of dependent
            # libraries to prevent clashes if an app using this library is
            # linked against other verions of our dependencies.
            set (VISIBILITY_MAP_COMMAND "-Wl,--version-script=${PROJECT_SOURCE_DIR}/src/build-scripts/hidesymbols.map")
        endif ()
    endif ()
    if (LLVM_STATIC AND ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND NOT DEBUGMODE)
        # Linux: When linking against LLVM statically, we can also hide
        # all its symbols to prevent clashes if OSL is linked against an
        # app that also embeds LLVM.
        set (VISIBILITY_MAP_COMMAND "-Wl,--version-script=${PROJECT_SOURCE_DIR}/src/build-scripts/hidesymbols.map")
    endif ()
endif ()

# Clang-specific options
if (CMAKE_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_APPLECLANG)
    # Disable some warnings for Clang, for some things that are too awkward
    # to change just for the sake of having no warnings.
    add_definitions ("-Wno-unused-function")
    add_definitions ("-Wno-overloaded-virtual")
    add_definitions ("-Wno-unneeded-internal-declaration")
    add_definitions ("-Wno-unused-private-field")
    add_definitions ("-Wno-tautological-compare")
    # disable warning about unused command line arguments
    add_definitions ("-Qunused-arguments")
    # Don't warn if we ask it not to warn about warnings it doesn't know
    add_definitions ("-Wunknown-warning-option")
    add_definitions ("-Wno-unknown-pragmas")
    # disable warning in flex-generated code
    add_definitions ("-Wno-null-conversion")
    add_definitions ("-Wno-error=strict-overflow")
    if (DEBUGMODE)
        add_definitions ("-Wno-unused-function -Wno-unused-variable")
    endif ()
    if (CLANG_VERSION_STRING VERSION_GREATER 3.5 OR
        APPLECLANG_VERSION_STRING VERSION_GREATER 6.1)
        add_definitions ("-Wno-unused-local-typedefs")
    endif ()
    if (CLANG_VERSION_STRING VERSION_EQUAL 3.9 OR CLANG_VERSION_STRING VERSION_GREATER 3.9)
        # Don't warn about using unknown preprocessor symbols in #if'set
        add_definitions ("-Wno-expansion-to-defined")
    endif ()
endif ()

# gcc specific options
if (CMAKE_COMPILER_IS_GNUCC AND NOT (CMAKE_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_APPLECLANG))
    if (NOT ${GCC_VERSION} VERSION_LESS 4.8)
        add_definitions ("-Wno-error=strict-overflow")
        # suppress a warning that Boost::Python hits in g++ 4.8
        add_definitions ("-Wno-error=unused-local-typedefs")
        add_definitions ("-Wno-unused-local-typedefs")
    endif ()
    if (NOT ${GCC_VERSION} VERSION_LESS 4.5)
        add_definitions ("-Wno-unused-result")
    endif ()
endif ()


## enable RTTI
##   NOTE: LLVM builds without RTTI by default so beware
##   if you find the need to turn this on, to use OSL in a
##   project that requires RTTI for example, you need to build
##   LLVM with RRTI, otherwise OSL classes extending LLVM ones
##   will cause linker errors.
set (ENABLERTTI OFF CACHE BOOL "Build with RTTI. Requires a LLVM build with RTTI enabled.")

if (NOT ENABLERTTI)
    if (MSVC)
        set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR-" )
    else ()
        set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti" )
    endif()

    # For gcc < 4.3, Boost needs some extra help detecting that we've
    # disabled RTTI.
    if ((CMAKE_COMPILER_IS_GNUCC AND GCC_VERSION VERSION_LESS 4.3)
        OR CMAKE_COMPILER_IS_CLANG)
        add_definitions ("-DBOOST_NO_RTTI")
        add_definitions ("-DBOOST_NO_TYPEID")
    endif ()
endif()

# Needed for static boost libraries on Windows
if (WIN32 AND Boost_USE_STATIC_LIBS)
    add_definitions ("-DBOOST_ALL_NO_LIB")
    add_definitions ("-DBOOST_THREAD_USE_LIB")
endif ()

# Needed to disable dllimport/dllexport for static library
if (BUILDSTATIC)
    add_definitions ("-DOSL_STATIC_LIBRARY")
endif()

set (USE_fPIC OFF CACHE BOOL "Build with -fPIC")
if (USE_fPIC)
    add_definitions ("-fPIC")
endif ()

# Try to detect if this is an OSX distro new enough that the system library
# is libc++.
if (EXISTS "/usr/lib/libc++.dylib")
    set (OSL_SYSTEM_HAS_LIBCPP ON)
endif ()


set (VERBOSE OFF CACHE BOOL "Print lots of messages while compiling")
set (OSL_BUILD_TESTS ON CACHE BOOL "Build the unit tests, testshade, testrender")
set (BUILDSTATIC OFF CACHE BOOL "Build static library instead of shared")
set (HIDE_SYMBOLS OFF CACHE BOOL "Hide symbols not in the public API")
set (USE_EXTERNAL_PUGIXML OFF CACHE BOOL
     "Use an externally built shared library version of the pugixml library")
set (PUGIXML_HOME "" CACHE STRING "Hint about where to find external PugiXML library")
if (WIN32)
    set (USE_LLVM_BITCODE OFF CACHE BOOL "Generate embedded LLVM bitcode")
else ()
    set (USE_LLVM_BITCODE ON CACHE BOOL "Generated embedded LLVM bitcode")
endif ()
set (USE_PARTIO ON CACHE BOOL "Use Partio if found")
set (PARTIO_HOME "" CACHE STRING "Search path for Partio components")
set (USE_FAST_MATH ON CACHE BOOL "Use fast math approximations (if no, then use system math library)")
set (USE_CPP11 ON CACHE BOOL "Compile in C++11 mode")
set (USE_CPP14 OFF CACHE BOOL "Compile in C++14 mode")
set (USE_LIBCPLUSPLUS OFF CACHE BOOL "Compile with clang libc++")
set (EXTRA_CPP_ARGS "" CACHE STRING "Extra C++ command line definitions")
set (USE_SIMD "" CACHE STRING "Use SIMD directives (0, sse2, sse3, sse4.1, sse4.2)")
set (OSL_NO_DEFAULT_TEXTURESYSTEM OFF CACHE BOOL "Do not use create a raw OIIO::TextureSystem")
set (OSL_BUILD_PLUGINS ON CACHE BOOL "Bool OSL plugins, for example OIIO plugin")
set (USE_CCACHE ON CACHE BOOL "Use ccache if found")
set (CODECOV OFF CACHE BOOL "Build code coverage tests")
option (USE_BOOST_WAVE "Use Boost Wave for C preprocessor (alternative is to use clang)" OFF)

# Use ccache if found
find_program (CCACHE_FOUND ccache)
if (CCACHE_FOUND AND USE_CCACHE)
    set_property (GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property (GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif ()

if (LLVM_NAMESPACE)
    add_definitions ("-DLLVM_NAMESPACE=${LLVM_NAMESPACE}")
endif ()

if (USE_EXTERNAL_PUGIXML)
    add_definitions ("-DUSE_EXTERNAL_PUGIXML")
endif ()

if (USE_FAST_MATH)
	add_definitions ("-DOSL_FAST_MATH=1")
else ()
	add_definitions ("-DOSL_FAST_MATH=0")
endif ()

set (USE_OIIO_STATIC ON CACHE BOOL "If OIIO is built static")
if (USE_OIIO_STATIC)
    add_definitions ("-DOIIO_STATIC_BUILD=1")
endif ()

if (OSL_NO_DEFAULT_TEXTURESYSTEM)
    add_definitions ("-DOSL_NO_DEFAULT_TEXTURESYSTEM=1")
endif ()

if (EXTRA_CPP_ARGS)
    message (STATUS "Extra C++ args: ${EXTRA_CPP_ARGS}")
    add_definitions ("${EXTRA_CPP_ARGS}")
endif()

if (USE_CPP11 OR USE_CPP14)
    if (USE_CPP14)
        message (STATUS "Building for C++14")
        add_definitions ("-std=c++14")
        set (OSL_CSTD_FLAGS ${OSL_CSTD_FLAGS} "-std=c++14")
    elseif (USE_CPP11)
        message (STATUS "Building for C++11")
        add_definitions ("-std=c++11")
        set (OSL_CSTD_FLAGS ${OSL_CSTD_FLAGS} "-std=c++11")
    endif ()
    if (CMAKE_COMPILER_IS_CLANG)
        # C++ >= 11 doesn't like 'register' keyword, which is in Qt headers
        add_definitions ("-Wno-deprecated-register")
    endif ()
else ()
    message (STATUS "Building for C++98!")
endif ()

if (USE_LIBCPLUSPLUS AND CMAKE_COMPILER_IS_CLANG)
    message (STATUS "Using libc++")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif ()

# Options common to gcc and clang
if (CODECOV AND (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANG))
    message (STATUS "Compiling for code coverage analysis")
    add_definitions ("-ftest-coverage -fprofile-arcs -O0 -DOSL_CODE_COVERAGE=1")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ftest-coverage -fprofile-arcs")
    set (CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -ftest-coverage -fprofile-arcs")
    set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -ftest-coverage -fprofile-arcs")
endif ()

if (DEFINED ENV{TRAVIS})
    add_definitions ("-DOSL_TRAVIS=1")
endif ()

set (OSL_SIMD_FLAGS "")
if (NOT USE_SIMD STREQUAL "")
    message (STATUS "Compiling with SIMD level ${USE_SIMD}")
    if (USE_SIMD STREQUAL "0")
        add_definitions ("-DOIIO_NO_SSE=1")
        set (OSL_SIMD_FLAGS ${OSL_SIMD_FLAGS} "-DOIIO_NO_SSE=1")
    else ()
        string (REPLACE "," ";" SIMD_FEATURE_LIST ${USE_SIMD})
        foreach (feature ${SIMD_FEATURE_LIST})
            if (VERBOSE)
                message (STATUS "SIMD feature: ${feature}")
            endif ()
            if (MSVC OR CMAKE_COMPILER_IS_INTEL)
                add_definitions ("/arch:${feature}")
                set (OSL_SIMD_FLAGS ${OSL_SIMD_FLAGS} "/arch:${feature}")
            else ()
                add_definitions ("-m${feature}")
                set (OSL_SIMD_FLAGS ${OSL_SIMD_FLAGS} "-m${feature}")
            endif ()
        endforeach()
    endif ()
endif ()

# Set the default namespace
if (OSL_NAMESPACE)
    add_definitions ("-DOSL_NAMESPACE=${OSL_NAMESPACE}")
endif ()
message(STATUS "Setting Namespace to: ${OSL_NAMESPACE}")

set (CMAKE_MODULE_PATH
     "${PROJECT_SOURCE_DIR}/src/cmake/modules"
     "${PROJECT_SOURCE_DIR}/src/cmake")

include (util_macros)
include (platform)
include (externalpackages)
include (flexbison)
include_directories (
    "${CMAKE_SOURCE_DIR}/src/include"
    "${CMAKE_BINARY_DIR}/include/OSL"
  )


###########################################################################
# Paths for install tree customization.  Note that relative paths are relative
# to CMAKE_INSTALL_PREFIX.
set (DEFAULT_BIN_INSTALL_DIR   "bin")
set (DEFAULT_LIB_INSTALL_DIR   "lib")
set (DEFAULT_INCLUDE_INSTALL_DIR "include/OSL")
if (UNIX AND NOT SELF_CONTAINED_INSTALL_TREE)
    # Try to be well-behaved and install into reasonable places according to
    # the "standard" unix directory heirarchy
    # TODO: Figure out how to get the correct python directory
    set (DEFAULT_PYLIB_INSTALL_DIR "lib/python/site-packages")
    set (DEFAULT_PYLIB3_INSTALL_DIR "lib/python3/site-packages")
    set (DEFAULT_DOC_INSTALL_DIR "share/doc/OSL")
    set (DEFAULT_MAN_INSTALL_DIR "share/man/man1")
else ()
    # Here is the "self-contained install tree" case: the expectation here is
    # that everything OSL related will go into its own directory, not into
    # some standard system heirarchy.
    set (DEFAULT_PYLIB_INSTALL_DIR "python")
    set (DEFAULT_PYLIB3_INSTALL_DIR "python3")
    set (DEFAULT_DOC_INSTALL_DIR "doc")
    set (DEFAULT_MAN_INSTALL_DIR "doc/man")
endif ()
if (EXEC_INSTALL_PREFIX)
    # Tack on an extra prefix to support multi-arch builds.
    set (DEFAULT_BIN_INSTALL_DIR   "${EXEC_INSTALL_PREFIX}/${DEFAULT_BIN_INSTALL_DIR}")
    set (DEFAULT_LIB_INSTALL_DIR   "${EXEC_INSTALL_PREFIX}/${DEFAULT_LIB_INSTALL_DIR}")
    set (DEFAULT_PYLIB_INSTALL_DIR "${EXEC_INSTALL_PREFIX}/${DEFAULT_PYLIB_INSTALL_DIR}")
    set (DEFAULT_PYLIB3_INSTALL_DIR "${EXEC_INSTALL_PREFIX}/${DEFAULT_PYLIB3_INSTALL_DIR}")
endif ()
# Set up cmake cache variables corresponding to the defaults deduced above, so
# that the user can override them as desired:
set (BIN_INSTALL_DIR ${DEFAULT_BIN_INSTALL_DIR} CACHE STRING
     "Install location for binaries (relative to CMAKE_INSTALL_PREFIX or absolute)")
set (LIB_INSTALL_DIR ${DEFAULT_LIB_INSTALL_DIR} CACHE STRING
     "Install location for libraries (relative to CMAKE_INSTALL_PREFIX or absolute)")
set (PYLIB_INSTALL_DIR ${DEFAULT_PYLIB_INSTALL_DIR} CACHE STRING
     "Install location for python libraries (relative to CMAKE_INSTALL_PREFIX or absolute)")
set (PYLIB3_INSTALL_DIR ${DEFAULT_PYLIB3_INSTALL_DIR} CACHE STRING
     "Install location for python3 libraries (relative to CMAKE_INSTALL_PREFIX or absolute)")
set (INCLUDE_INSTALL_DIR ${DEFAULT_INCLUDE_INSTALL_DIR} CACHE STRING
     "Install location of header files (relative to CMAKE_INSTALL_PREFIX or absolute)")
set (DOC_INSTALL_DIR ${DEFAULT_DOC_INSTALL_DIR} CACHE STRING
     "Install location for documentation (relative to CMAKE_INSTALL_PREFIX or absolute)")
if (UNIX)
    set (MAN_INSTALL_DIR ${DEFAULT_MAN_INSTALL_DIR} CACHE STRING
         "Install location for manual pages (relative to CMAKE_INSTALL_PREFIX or absolute)")
endif()
set (PLUGIN_SEARCH_PATH "" CACHE STRING "Default plugin search path")
set (INSTALL_DOCS ON CACHE BOOL "Install documentation")


if (APPLE)
    set (MACOSX_RPATH ON)
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
    if(NOT LINKSTATIC)
        add_definitions (-DBOOST_ALL_DYN_LINK)
        add_definitions (-DOPENEXR_DLL)
    endif ()
endif (MSVC)


# We want CTest for testing
# N.B. This needs to be added before any of the subdirectories, or
# their add_test commands will not register.
include (CTest)


# Tell CMake to process the sub-directories
add_subdirectory (src/liboslcomp)
add_subdirectory (src/liboslquery)
add_subdirectory (src/liboslexec)
add_subdirectory (src/liboslnoise)

add_subdirectory (src/oslc)
add_subdirectory (src/shaders)
add_subdirectory (src/oslinfo)

if (OSL_BUILD_TESTS)
add_subdirectory (src/testshade)
add_subdirectory (src/testrender)
endif ()

if (OSL_BUILD_PLUGINS)
add_subdirectory (src/osl.imageio)
endif ()

add_subdirectory (src/include)
add_subdirectory (src/doc)



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
    parse_arguments (_ats "LABEL" "" ${ARGN})
    # If there was a FOUNDVAR param specified and that variable name is
    # not defined, mark the test as broken.
    if (_ats_FOUNDVAR AND NOT ${_ats_FOUNDVAR})
        set (_ats_LABEL "broken")
    endif ()
    # Add the tests if all is well.
    set (ALL_TEST_LIST "")
    foreach (_testname ${_ats_DEFAULT_ARGS})
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

        file (MAKE_DIRECTORY "${_testdir}")

        # Run the test unoptimized, unless it's a "render_*" or "oslinfo_*"
        # test, which we don't bother testing unoptimized.
        if (NOT _testname MATCHES "^render" AND
            NOT _testname MATCHES "^oslinfo" AND
            NOT _testname MATCHES "^getattribute-shader" AND
            NOT EXISTS "${_testsrcdir}/OPTIMIZEONLY")
          set (_env TESTSHADE_OPT=0 OPENIMAGEIOHOME=${OPENIMAGEIOHOME})
          add_test ( NAME ${_testname}
                     COMMAND env ${_env} ${_runtest} )
        endif ()
        # Run the same test again with aggressive -O2 runtime
        # optimization, triggered by setting TESTSHADE_OPT env variable
        set (_env TESTSHADE_OPT=2 OPENIMAGEIOHOME=${OPENIMAGEIOHOME})
        set (ALL_TEST_LIST "${ALL_TEST_LIST} ${_testname}")
        add_test ( NAME ${_testname}.opt
                   COMMAND env ${_env} ${_runtest} )
    endforeach ()
    if (VERBOSE)
        message (STATUS "Added tests: ${ALL_TEST_LIST}")
    endif ()
endmacro ()

if (OSL_BUILD_TESTS)
# List all the individual testsuite tests here, except those that need
# special installed tests.
TESTSUITE ( and-or-not-synonyms aastep arithmetic array array-derivs array-range
            blackbody blendmath breakcont
            bug-array-heapoffsets
            bug-locallifetime bug-outputinit bug-param-duplicate bug-peep
            cellnoise closure closure-array color comparison
            compile-buffer
            component-range const-array-params const-array-fill
            debugnan debug-uninit
            derivs derivs-muldiv-clobber
            draw_string
            error-dupes exit exponential
            function-earlyreturn function-simple function-outputelem
            geomath getattribute-camera getattribute-shader
            getsymbol-nonheap gettextureinfo
            group-outputs groupstring
            hash hashnoise hex hyperb
            ieee_fp if incdec initops intbits isconnected isconstant
            layers layers-Ciassign layers-entry layers-lazy
            layers-nonlazycopy layers-repeatedoutputs
            linearstep
            logic loop matrix message
            mergeinstances-nouserdata mergeinstances-vararray
            metadata-braces miscmath missing-shader
            noise noise-cell
            noise-gabor noise-gabor2d-filter noise-gabor3d-filter
            noise-perlin noise-uperlin noise-simplex noise-usimplex
            pnoise pnoise-cell pnoise-gabor pnoise-perlin pnoise-uperlin
            oslc-comma oslc-D
            oslc-err-arrayindex oslc-err-closuremul
            oslc-err-format oslc-err-intoverflow
            oslc-err-noreturn oslc-err-notfunc
            oslc-err-outputparamvararray oslc-err-paramdefault
            oslc-err-struct-array-init oslc-err-struct-dup
            oslc-warn-commainit
            oslc-variadic-macro
            oslc-version
            oslinfo-arrayparams oslinfo-colorctrfloat
            oslinfo-metadata oslinfo-noparams
            osl-imageio
            printf-whole-array
            raytype raytype-specialized reparam
            render-background render-bumptest
            render-cornell render-furnace-diffuse
            render-microfacet render-oren-nayar render-veachmis render-ward
            select shortcircuit spline splineinverse
            spline-boundarybug spline-derivbug
            string
            struct struct-array struct-array-mixture
            struct-err struct-init-copy struct-layers
            struct-return struct-with-array
            struct-nested struct-nested-assign struct-nested-deep
            ternary
            testshade-expr
            texture-alpha texture-blur texture-connected-options
            texture-derivs texture-errormsg
            texture-firstchannel texture-interp
            texture-missingcolor texture-simple
            texture-smallderivs texture-swirl texture-udim
            texture-width texture-withderivs texture-wrap
            trailing-commas
            transitive-assign
            transform transformc trig typecast
            unknown-instruction
            vararray-connect vararray-default
            vararray-deserialize vararray-param
            vecctr vector
            wavelength_color xml )

# Only run field3d-related tests if the local OIIO was built with f3d support.
EXECUTE_PROCESS ( COMMAND ${OPENIMAGEIO_BIN}/oiiotool --help
                  OUTPUT_VARIABLE oiiotool_help )
if (oiiotool_help MATCHES "field3d")
    TESTSUITE ( texture-field3d )
endif()

# Only run pointcloud tests if Partio is found
if (PARTIO_FOUND)
    TESTSUITE ( pointcloud pointcloud-fold )
endif ()

endif (OSL_BUILD_TESTS)



#########################################################################
# Packaging
set (CPACK_PACKAGE_VERSION_MAJOR ${OSL_LIBRARY_VERSION_MAJOR})
set (CPACK_PACKAGE_VERSION_MINOR ${OSL_LIBRARY_VERSION_MINOR})
set (CPACK_PACKAGE_VERSION_PATCH ${OSL_LIBRARY_VERSION_PATCH})
# "Vendor" is only used in copyright notices, so we use the same thing that
# the rest of the copyright notices say.
set (CPACK_PACKAGE_VENDOR "Sony Pictures Imageworks")
set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "OpenShadingLanguage is...")
set (CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/src/doc/Description.txt")
set (CPACK_PACKAGE_FILE_NAME OSL-${OSL_LIBRARY_VERSION_MAJOR}.${OSL_LIBRARY_VERSION_MINOR}.${OSL_LIBRARY_VERSION_PATCH}-${platform})
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
set (CPACK_SOURCE_PACKAGE_FILE_NAME OSL-${OSL_LIBRARY_VERSION_MAJOR}.${OSL_LIBRARY_VERSION_MINOR}.${OSL_LIBRARY_VERSION_PATCH}-source)
#set (CPACK_SOURCE_STRIP_FILES ...FIXME...)
set (CPACK_SOURCE_IGNORE_FILES ".*~")
include (CPack)
