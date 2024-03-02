# Copyright Contributors to the Open Shading Language project.
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


set (OSL_TEST_BIG_TIMEOUT 800 CACHE STRING "Timeout for tests that take a long time")


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
    file (MAKE_DIRECTORY "${testdir}")
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

    # We need to suppress the leak sanitizer for "texture" tests, because of a
    # minor leak inside OIIO (140 bytes! -- inconsequential) but which we
    # can't use the suppressions list becuase the leaks come from calls made
    # inside JITed code for which there is not a good stack trace. Any real
    # leaks from OIIO or its dependencies we assume will be caught by lsan on
    # the OIIO side, where there is no JIT to complicate things.
    if (${testname} MATCHES "^texture" OR ${testname} MATCHES "gettextureinfo")
        list (APPEND _tst_ENV ASAN_OPTIONS=detect_leaks=0:$ENV{ASAN_OPTIONS})
    endif ()

    add_test ( NAME ${testname} COMMAND ${_tst_COMMAND} )
    # message ("Test -- env ${_tst_ENV} cmd ${_tst_COMMAND}")
    set_tests_properties (${testname} PROPERTIES ENVIRONMENT "${_tst_ENV}" )
    # Certain tests are already internally multi-threaded, so to keep them
    # from piling up together, allocate a few cores each.
    if (${testname} MATCHES "^render-")
        set_tests_properties (${testname} PROPERTIES LABELS render
                              PROCESSORS 4 COST 10 TIMEOUT ${OSL_TEST_BIG_TIMEOUT})
    endif ()
    # For debug builds, render tests are very slow, so give them a longer timeout
    if (${testname} MATCHES "render-" AND ${CMAKE_BUILD_TYPE} STREQUAL Debug)
        set_tests_properties (${testname} PROPERTIES TIMEOUT ${OSL_TEST_BIG_TIMEOUT})
    endif ()
    # Some labeling for fun
    if (${testname} MATCHES "^texture")
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
            set_property (TEST ${testname} APPEND PROPERTY ENVIRONMENT LD_LIBRARY_PATH=${CUDA_TOOLKIT_ROOT_DIR}/lib64:$ENV{LD_LIBRARY_PATH})
        endif()
    endif ()
    if (${testname} MATCHES "-reg")
        # These are batched shading regression tests. Some are quite
        # long, so give them a higher cost and timeout.
        set_tests_properties (${testname} PROPERTIES LABELS batchregression
                              COST 15 TIMEOUT ${OSL_TEST_BIG_TIMEOUT})
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
    set (test_all_rs_bitcode $ENV{TESTSUITE_RS_BITCODE})
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
        if (NOT _testname MATCHES "optix"
            AND NOT EXISTS "${_testsrcdir}/NOSCALAR"
            AND NOT EXISTS "${_testsrcdir}/BATCHED_REGRESSION"
            AND NOT EXISTS "${_testsrcdir}/OPTIMIZEONLY")
            add_one_testsuite ("${_testname}" "${_testsrcdir}"
                               ENV TESTSHADE_OPT=0 )
        endif ()
        # Run the same test again with aggressive -O2 runtime
        # optimization, triggered by setting TESTSHADE_OPT env variable.
        # Skip OptiX-only tests and those with a NOOPTIMIZE marker file.
        if (NOT _testname MATCHES "optix"
            AND NOT EXISTS "${_testsrcdir}/NOSCALAR"
            AND NOT EXISTS "${_testsrcdir}/BATCHED_REGRESSION"
            AND NOT EXISTS "${_testsrcdir}/NOOPTIMIZE")
            add_one_testsuite ("${_testname}.opt" "${_testsrcdir}"
                               ENV TESTSHADE_OPT=2 )
        endif ()
        # Run the same test again with aggressive -O2 runtime
        # optimization, triggered by setting TESTSHADE_OPT env variable.
        # Skip OptiX-only tests and those with a NOOPTIMIZE marker file.
        if (NOT _testname MATCHES "optix"
            AND NOT EXISTS "${_testsrcdir}/NOSCALAR"
            AND NOT EXISTS "${_testsrcdir}/BATCHED_REGRESSION"
            AND NOT EXISTS "${_testsrcdir}/NOOPTIMIZE"
            AND NOT EXISTS "${_testsrcdir}/NORSBITCODE")
            add_one_testsuite ("${_testname}.opt.rs_bitcode" "${_testsrcdir}"
                               ENV TESTSHADE_OPT=2 TESTSHADE_RS_BITCODE=1)
        endif ()
        # When building for OptiX support, also run it in OptiX mode
        # if there is an OPTIX marker file in the directory.
        # If an environment variable $TESTSUITE_OPTIX is nonzero, then
        # run all tests with OptiX, even if there's no OPTIX marker.
        if (OSL_USE_OPTIX
            AND (EXISTS "${_testsrcdir}/OPTIX" OR test_all_optix OR _testname MATCHES "optix")
            AND NOT EXISTS "${_testsrcdir}/NOOPTIX"
            AND NOT EXISTS "${_testsrcdir}/NOOPTIX-FIXME"
            AND NOT EXISTS "${_testsrcdir}/BATCHED_REGRESSION")
            # Unoptimized
            if (NOT EXISTS "${_testsrcdir}/OPTIMIZEONLY")
                add_one_testsuite ("${_testname}.optix" "${_testsrcdir}"
                                   ENV TESTSHADE_OPT=0 TESTSHADE_OPTIX=1 )
            endif ()
            # and optimized
            add_one_testsuite ("${_testname}.optix.opt" "${_testsrcdir}"
                               ENV TESTSHADE_OPT=2 TESTSHADE_OPTIX=1 )
            if (NOT EXISTS "${_testsrcdir}/NOFUSED")
              # and fused
              add_one_testsuite ("${_testname}.optix.fused" "${_testsrcdir}"
                                 ENV TESTSHADE_OPT=2 TESTSHADE_OPTIX=1 TESTSHADE_FUSED=1 )
            endif()
        endif ()

        if (OSL_BUILD_BATCHED)
            # When building for Batched support, also run it in Batched mode
            # if there is an BATCHED marker file in the directory.
            # If an environment variable $TESTSUITE_BATCHED is nonzero, then
            # run all tests in Batched mode, even if there's no BATCHED marker.
            if ((EXISTS "${_testsrcdir}/BATCHED" OR test_all_batched OR _testname MATCHES "batched")
                AND NOT EXISTS "${_testsrcdir}/BATCHED_REGRESSION"
                AND NOT EXISTS "${_testsrcdir}/NOBATCHED"
                AND NOT EXISTS "${_testsrcdir}/NOBATCHED-FIXME"
                AND NOT EXISTS "${_testsrcdir}/OPTIMIZEONLY")
                add_one_testsuite ("${_testname}.batched" "${_testsrcdir}"
                                   ENV TESTSHADE_OPT=0 TESTSHADE_BATCHED=1 )
            endif ()

            if ((EXISTS "${_testsrcdir}/BATCHED" OR test_all_batched OR _testname MATCHES "batched")
                AND NOT EXISTS "${_testsrcdir}/BATCHED_REGRESSION"
                AND NOT EXISTS "${_testsrcdir}/NOBATCHED"
                AND NOT EXISTS "${_testsrcdir}/NOBATCHED-FIXME"
                AND NOT EXISTS "${_testsrcdir}/NOOPTIMIZE")
                add_one_testsuite ("${_testname}.batched.opt" "${_testsrcdir}"
                                   ENV TESTSHADE_OPT=2 TESTSHADE_BATCHED=1 )
            endif ()

            # When building for Batched support, optionally run a regression test
            # if there is an BATCHED_REGRESSION marker file in the directory.
            if ((EXISTS "${_testsrcdir}/BATCHED_REGRESSION")
                AND NOT EXISTS "${_testsrcdir}/NOOPTIMIZE")
                # optimized for right now
                add_one_testsuite ("${_testname}.regress.batched.opt" "${_testsrcdir}"
                                   ENV TESTSHADE_OPT=2 OSL_REGRESSION_TEST=BATCHED )
            endif ()
            if ((EXISTS "${_testsrcdir}/BATCHED_REGRESSION")
                AND EXISTS "${_testsrcdir}/NOOPTIMIZE")
                # not optimized
                add_one_testsuite ("${_testname}.regress.batched" "${_testsrcdir}"
                                   ENV TESTSHADE_OPT=0 OSL_REGRESSION_TEST=BATCHED )
            endif ()
        endif ()

        # if there is an RS_BITCODE marker file in the directory.
        if ((EXISTS "${_testsrcdir}/RS_BITCODE" OR test_all_rs_bitcode)
            AND NOT EXISTS "${_testsrcdir}/BATCHED_REGRESSION"
            AND NOT EXISTS "${_testsrcdir}/RS_BITCODE_REGRESSION"
            AND NOT EXISTS "${_testsrcdir}/NOOPTIMIZE")
            add_one_testsuite ("${_testname}.rsbitcode.opt" "${_testsrcdir}"
                                ENV TESTSHADE_OPT=2 TESTSHADE_RS_BITCODE=1 )
        endif ()

        # if there is an RS_BITCODE_REGRESSION marker file in the directory.
        if (EXISTS "${_testsrcdir}/RS_BITCODE_REGRESSION" OR (
                test_all_rs_bitcode AND EXISTS "${_testsrcdir}/BATCHED_REGRESSION")
            AND NOT EXISTS "${_testsrcdir}/NOOPTIMIZE")
            # optimized for right now
            add_one_testsuite ("${_testname}.regress.rsbitcode.opt" "${_testsrcdir}"
                                ENV TESTSHADE_OPT=2 OSL_REGRESSION_TEST=RS_BITCODE )
        endif ()

    endforeach ()
    message (VERBOSE "Added tests: ${ALL_TEST_LIST}")
endmacro ()

macro (osl_add_all_tests)
    # List all the individual testsuite tests here, except those that need
    # special installed tests.
    TESTSUITE ( aastep allowconnect-err andor-reg and-or-not-synonyms
                arithmetic area-reg arithmetic-reg
                array array-reg array-copy-reg array-derivs array-range
                array-aassign array-assign-reg array-length-reg
                bitwise-and-reg bitwise-or-reg bitwise-shl-reg  bitwise-shr-reg bitwise-xor-reg
                blackbody blackbody-reg blendmath breakcont breakcont-reg
                bug-array-heapoffsets bug-locallifetime bug-outputinit
                bug-param-duplicate bug-peep bug-return
                calculatenormal-reg
                cellnoise closure closure-array closure-layered closure-parameters closure-zero closure-conditional
                color color-reg colorspace comparison
                complement-reg compile-buffer compassign-bool compassign-reg
                component-range
                control-flow-reg connect-components
                const-array-params const-array-fill
                debugnan debug-uninit
                derivs derivs-muldiv-clobber
                draw_string
                error-dupes error-serialized
                example-deformer
                example-batched-deformer
                exit exponential
                filterwidth-reg
                for-reg format-reg fprintf
                function-earlyreturn function-simple function-outputelem
                function-overloads function-redef
                geomath getattribute-camera getattribute-shader getattribute-shading
                getsymbol-nonheap gettextureinfo gettextureinfo-reg
                gettextureinfo-udim gettextureinfo-udim-reg
                globals-needed
                group-outputs groupdata-opt groupstring
                hash hashnoise hex hyperb
                ieee_fp ieee_fp-reg if if-reg incdec initlist
                initops initops-instance-clash
                intbits isconnected
                isconstant
                layers layers-Ciassign layers-entry layers-lazy layers-lazyerror
                layers-nonlazycopy layers-repeatedoutputs
                length-reg linearstep
                logic loop luminance-reg
                matrix matrix-reg matrix-arithmetic-reg
                matrix-compref-reg max-reg message message-no-closure message-reg
                mergeinstances-duplicate-entrylayers
                mergeinstances-nouserdata mergeinstances-vararray
                metadata-braces min-reg miscmath missing-shader
                mix-reg
                named-components
                nestedloop-reg
                noise noise-cell
                noise-gabor noise-gabor2d-filter noise-gabor3d-filter
                noise-gabor-reg
                noise-generic
                noise-perlin noise-simplex
                noise-reg
                normalize-reg
                pnoise pnoise-cell pnoise-gabor
                pnoise-generic pnoise-perlin
                pnoise-reg
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
                oslc-err-type-as-variable
                oslc-err-unknown-ctr
                oslc-literalfold
                oslc-pragma-warnerr
                oslc-warn-commainit
                oslc-variadic-macro
                oslc-version
                oslinfo-arrayparams oslinfo-colorctrfloat
                oslinfo-metadata oslinfo-noparams
                osl-imageio
                paramval-floatpromotion
                pragma-nowarn
                printf-reg
                printf-whole-array
                raytype raytype-reg raytype-specialized regex-reg
                reparam reparam-arrays testoptix-reparam
                render-background render-bumptest
                render-cornell render-furnace-diffuse
                render-mx-furnace-burley-diffuse
                render-mx-furnace-oren-nayar
                render-mx-furnace-sheen
                render-mx-burley-diffuse
                render-mx-dielectric render-mx-dielectric-glass
                render-mx-conductor
                render-mx-generalized-schlick render-mx-generalized-schlick-glass
                render-mx-layer
                render-mx-sheen
                render-microfacet render-oren-nayar
                render-uv render-veachmis render-ward
                render-raytypes
                select select-reg shaderglobals shortcircuit
                smoothstep-reg
                spline spline-reg splineinverse splineinverse-ident
                splineinverse-knots-ascend-reg splineinverse-knots-descend-reg
                spline-boundarybug spline-derivbug
                split-reg
                string string-reg
                struct struct-array struct-array-mixture
                struct-err struct-init-copy
                struct-isomorphic-overload struct-layers
                struct-operator-overload struct-return struct-with-array
                struct-nested struct-nested-assign struct-nested-deep
                ternary
                testshade-expr
		test-fmt-arrays test-fmt-fileprint
		test-fmt-cxpf  test-fmt-noise test-fmt-matrixcolor 
                test-fmt-stpf test-fmt-errorwarning test-fmt-errorwarning-repeats
                texture-alpha texture-alpha-derivs
                texture-blur texture-connected-options
                texture-derivs texture-environment texture-errormsg
                texture-environment-opts-reg
                texture-firstchannel texture-interp
                texture-missingalpha texture-missingcolor texture-opts-reg texture-simple
                texture-smallderivs texture-swirl texture-udim
                texture-width texture-withderivs texture-wrap
                trace-reg
                trailing-commas
                transcendental-reg
                transitive-assign
                transform transform-reg transformc transformc-reg trig trig-reg
                typecast
                unknown-instruction
                userdata userdata-partial userdata-custom userdata-passthrough
                vararray-connect vararray-default
                vararray-deserialize vararray-param
                vecctr vector vector-reg
                wavelength_color wavelength_color-reg Werror xml xml-reg )

    # Only run the ocio test if the OIIO we are using has OCIO support
    if (OpenImageIO_HAS_OpenColorIO)
        TESTSUITE ( ocio )
    endif ()

    # Add tests that require the Python bindings if we built them.
    # We also exclude these tests if this is a sanitizer build, because the
    # Python interpreter itself won't be linked with the right asan
    # libraries to run correctly.
    if (USE_PYTHON AND NOT SANITIZE)
        TESTSUITE ( python-oslquery )
    endif ()

    # Only run openvdb-related tests if the local OIIO has openvdb support.
    execute_process ( COMMAND ${OpenImageIO_LIB_DIR}/../bin/oiiotool --help
                      OUTPUT_VARIABLE oiiotool_help )
    if (oiiotool_help MATCHES "openvdb")
        TESTSUITE ( texture3d texture3d-opts-reg )
    endif()

    # Only run pointcloud tests if Partio is found
    if (partio_FOUND)
        TESTSUITE ( pointcloud pointcloud-fold )
    endif ()

    # Only run the OptiX tests if OptiX and CUDA are found
    if (OPTIX_FOUND AND CUDA_FOUND)
        TESTSUITE ( testoptix testoptix-noise example-cuda)
    endif ()

    # Some regression tests have a lot of combinations and may need more time to finish
    if (OSL_BUILD_BATCHED)
        set_tests_properties (arithmetic-reg.regress.batched.opt
                              PROPERTIES TIMEOUT ${OSL_TEST_BIG_TIMEOUT})
        set_tests_properties (transform-reg.regress.batched.opt
                              PROPERTIES TIMEOUT ${OSL_TEST_BIG_TIMEOUT})
    endif ()
    set_tests_properties (matrix-reg.regress.rsbitcode.opt
                          PROPERTIES TIMEOUT ${OSL_TEST_BIG_TIMEOUT})
endmacro()
