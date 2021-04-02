# Copyright Contributors to the Open Shading Languge project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage/

include (CTest)

# Make a build/platform/testsuite directory, and copy the master runtest.py
# there. The rest is up to the tests themselves.
file (MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/testsuite")
file (COPY "${CMAKE_CURRENT_SOURCE_DIR}/testsuite/common"
      DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/testsuite")
add_custom_command (OUTPUT "${CMAKE_BINARY_DIR}/testsuite/runtest.py"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${CMAKE_SOURCE_DIR}/testsuite/runtest.py"
                        "${CMAKE_BINARY_DIR}/testsuite/runtest.py"
                    MAIN_DEPENDENCY "${CMAKE_SOURCE_DIR}/testsuite/runtest.py")
add_custom_target ( CopyFiles ALL DEPENDS "${CMAKE_BINARY_DIR}/testsuite/runtest.py" )

# add_one_testsuite() - set up one testsuite entry
#
# Usage:
#   add_one_testsuite ( testname
#                  testsrcdir - Current test directory in ${CMAKE_SOURCE_DIR}
#                  testdir    - Current test sandbox in ${CMAKE_BINARY_DIR}
#                  [ENV env1=val1 env2=val2 ... ]  - env vars to set
#                  [COMMAND cmd...] - optional override of launch command
#                 )
#
macro (add_one_testsuite testname testsrcdir)
    cmake_parse_arguments (_tst "" "" "ENV;COMMAND" ${ARGN})
    set (testsuite "${CMAKE_SOURCE_DIR}/testsuite")
    set (testdir "${CMAKE_BINARY_DIR}/testsuite/${testname}")
    if (NOT _tst_COMMAND)
        set (_tst_COMMAND ${Python_EXECUTABLE} "${testsuite}/runtest.py" ${testdir})
        if (MSVC_IDE)
            list (APPEND _tst_COMMAND --devenv-config $<CONFIGURATION>
                                      --solution-path "${CMAKE_BINARY_DIR}" )
        endif ()
    endif ()
    list (APPEND _tst_ENV
              OpenImageIO_ROOT=${OpenImageIO_ROOT}
              OSL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
              OSL_BUILD_DIR=${CMAKE_BINARY_DIR}
              OSL_TESTSUITE_ROOT=${testsuite}
              OSL_TESTSUITE_SRC=${testsrcdir}
              OSL_TESTSUITE_CUR=${testdir}
         )
    file (MAKE_DIRECTORY "${testdir}")
    add_test ( NAME ${testname} COMMAND ${_tst_COMMAND} )
    # message ("Test -- env ${_tst_ENV} cmd ${_tst_COMMAND}")
    set_tests_properties (${testname} PROPERTIES ENVIRONMENT "${_tst_ENV}" )
    # Certain tests are already internally multi-threaded, so to keep them
    # from piling up together, allocate a few cores each.
    if (${testname} MATCHES "^render-")
        set_tests_properties (${testname} PROPERTIES LABELS render
                              PROCESSORS 4 COST 10)
    endif ()
    # Some labeling for fun
    if (${testname} MATCHES "^texture-")
        set_tests_properties (${testname} PROPERTIES LABELS texture
                              PROCESSORS 2 COST 4)
    endif ()
    if (${testname} MATCHES "noise")
        set_tests_properties (${testname} PROPERTIES LABELS noise
                              PROCESSORS 2 COST 4)
    endif ()
    if (${testname} MATCHES "optix")
        set_tests_properties (${testname} PROPERTIES LABELS optix)
        if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "10.0")
            # Make sure libnvrtc-builtins.so is reachable
            set_property (TEST ${testname} APPEND PROPERTY ENVIRONMENT LD_LIBRARY_PATH=${CUDA_TOOLKIT_ROOT_DIR}/lib64)
        endif()
    endif ()
endmacro ()


macro ( TESTSUITE )
    cmake_parse_arguments (_ats "" "LABEL;FOUNDVAR;TESTNAME" "" ${ARGN})
    # If there was a FOUNDVAR param specified and that variable name is
    # not defined, mark the test as broken.
    if (_ats_FOUNDVAR AND NOT ${_ats_FOUNDVAR})
        set (_ats_LABEL "broken")
    endif ()
    set (test_all_optix $ENV{TESTSUITE_OPTIX})
    set (test_all_batched $ENV{TESTSUITE_BATCHED})
    # Add the tests if all is well.
    set (ALL_TEST_LIST "")
    set (_testsuite "${CMAKE_SOURCE_DIR}/testsuite")
    foreach (_testname ${_ats_UNPARSED_ARGUMENTS})
        set (_testsrcdir "${_testsuite}/${_testname}")
        if (_ats_TESTNAME)
            set (_testname "${_ats_TESTNAME}")
        endif ()
        if (_ats_LABEL MATCHES "broken")
            set (_testname "${_testname}-broken")
        endif ()

        set (ALL_TEST_LIST "${ALL_TEST_LIST} ${_testname}")

        # Run the test unoptimized, unless it matches a few patterns that
        # we don't test unoptimized (or has an OPTIMIZEONLY marker file).
        if (NOT _testname MATCHES "^getattribute-shader" AND
            NOT _testname MATCHES "optix" AND
            NOT EXISTS "${_testsrcdir}/OPTIMIZEONLY")
            add_one_testsuite ("${_testname}" "${_testsrcdir}"
                               ENV TESTSHADE_OPT=0 )
        endif ()
        # Run the same test again with aggressive -O2 runtime
        # optimization, triggered by setting TESTSHADE_OPT env variable.
        # Skip OptiX-only tests and those with a NOOPTIMIZE marker file.
        if (NOT _testname MATCHES "optix"
                AND NOT EXISTS "${_testsrcdir}/NOOPTIMIZE")
            add_one_testsuite ("${_testname}.opt" "${_testsrcdir}"
                               ENV TESTSHADE_OPT=2 )
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
                add_one_testsuite ("${_testname}.optix" "${_testsrcdir}"
                                   ENV TESTSHADE_OPT=0 TESTSHADE_OPTIX=1 )
            endif ()
            # and optimized
            add_one_testsuite ("${_testname}.optix.opt" "${_testsrcdir}"
                               ENV TESTSHADE_OPT=2 TESTSHADE_OPTIX=1 )
        endif ()
        
        # When building for Batched support, also run it in Batched mode
        # if there is an BATCHED marker file in the directory.
        # If an environment variable $TESTSUITE_BATCHED is nonzero, then
        # run all tests in Batched mode, even if there's no BATCHED marker.
        if ((EXISTS "${_testsrcdir}/BATCHED" OR test_all_batched OR _testname MATCHES "batched")
            AND NOT EXISTS "${_testsrcdir}/NOBATCHED"
            AND NOT EXISTS "${_testsrcdir}/NOBATCHED-FIXME")
            # optimized for right now
            add_one_testsuite ("${_testname}.batched.opt" "${_testsrcdir}"
                               ENV TESTSHADE_OPT=2 TESTSHADE_BATCHED=1 )
        endif ()
    endforeach ()
    if (VERBOSE)
        message (STATUS "Added tests: ${ALL_TEST_LIST}")
    endif ()
endmacro ()


macro (osl_add_all_tests)
    # List all the individual testsuite tests here, except those that need
    # special installed tests.
    TESTSUITE ( aastep allowconnect-err and-or-not-synonyms arithmetic
                arithmetic-cov
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
                example-deformer
                exit exponential
                fprintf
                function-earlyreturn function-simple function-outputelem
                function-overloads function-redef
                geomath getattribute-camera getattribute-shader
                getsymbol-nonheap gettextureinfo
                group-outputs groupstring
                hash hashnoise hex hyperb
                ieee_fp if incdec initlist initops intbits isconnected isconstant
                layers layers-Ciassign layers-entry layers-lazy layers-lazyerror
                layers-nonlazycopy layers-repeatedoutputs
                linearstep
                logic loop matrix message
                mergeinstances-duplicate-entrylayers
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
                oslc-pragma-warnerr
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
                select shaderglobals shortcircuit 
                spline splineinverse splineinverse-ident
                spline-boundarybug spline-derivbug
                string
                struct struct-array struct-array-mixture
                struct-err struct-init-copy
                struct-isomorphic-overload struct-layers
                struct-operator-overload struct-return struct-with-array
                struct-nested struct-nested-assign struct-nested-deep
                ternary
                testshade-expr
                texture-alpha texture-alpha-derivs
                texture-blur texture-connected-options
                texture-derivs texture-errormsg
                texture-firstchannel texture-interp
                texture-missingalpha texture-missingcolor texture-simple
                texture-smallderivs texture-swirl texture-udim
                texture-width texture-withderivs texture-wrap
                trailing-commas
                transitive-assign
                transform transformc trig typecast
                unknown-instruction
                userdata userdata-passthrough
                vararray-connect vararray-default
                vararray-deserialize vararray-param
                vecctr vector
                wavelength_color Werror xml )

    # Add tests that require the Python bindings if we built them.
    # We also exclude these tests if this is a sanitizer build on Linux,
    # because the Python interpreter itself won't be linked with the right asan
    # libraries to run correctly.
    if (USE_PYTHON AND NOT SANITIZE_ON_LINUX)
        TESTSUITE ( python-oslquery )
    endif ()

    # Only run field3d-related tests if the local OIIO was built with f3d support.
    execute_process ( COMMAND ${OPENIMAGEIO_BIN} --help
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
        TESTSUITE ( testoptix testoptix-noise example-cuda)
    endif ()

endmacro()
