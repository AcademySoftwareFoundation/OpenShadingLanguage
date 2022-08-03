# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


function ( EMBED_LLVM_BITCODE_IN_CPP src_list suffix output_name list_to_append_cpp extra_clang_args)

    if (VERBOSE)
        message (STATUS "EMBED_LLVM_BITCODE_IN_CPP src_list=${src_list}")
    endif ()

    foreach ( src ${src_list} )
        get_filename_component ( src_we ${src} NAME_WE )
        set ( src_asm "${CMAKE_CURRENT_BINARY_DIR}/${src_we}${suffix}.s" )
        set ( src_bc "${CMAKE_CURRENT_BINARY_DIR}/${src_we}${suffix}.bc" )
        if (VERBOSE)
            message (STATUS "EMBED_LLVM_BITCODE_IN_CPP in=${src}")
            message (STATUS "EMBED_LLVM_BITCODE_IN_CPP asm=${src_asm}")
            message (STATUS "EMBED_LLVM_BITCODE_IN_CPP bc=${src_bc}")
        endif ()
        list ( APPEND src_bc_list ${src_bc} )

        get_property (CURRENT_DEFINITIONS DIRECTORY PROPERTY COMPILE_DEFINITIONS)
        if (VERBOSE)
            message (STATUS "Current #defines are ${CURRENT_DEFINITIONS}")
        endif ()
        foreach (def ${CURRENT_DEFINITIONS})
            set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} "-D${def}")
        endforeach()
        set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} ${SIMD_COMPILE_FLAGS} ${CSTD_FLAGS} ${TOOLCHAIN_FLAGS})
        # Avoid generating __dso_handle external global
        set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} "-fno-use-cxa-atexit")

        # Figure out what program we will use to make the bitcode.
        if (NOT LLVM_BC_GENERATOR)
            find_program (LLVM_BC_GENERATOR NAMES "clang++"
                        PATHS "${LLVM_DIRECTORY}/bin" "${LLVM_DIRECTORY}/tools/llvm"
                        NO_CMAKE_PATH NO_DEFAULT_PATH NO_CMAKE_SYSTEM_PATH
                        NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_ENVIRONMENT_PATH)
        endif ()

        if (NOT LLVM_BC_GENERATOR)
            message (FATAL_ERROR "You must have a valid llvm bitcode generator (clang++) somewhere.")
        endif ()
        if (VERBOSE)
            message (STATUS "Using LLVM_BC_GENERATOR ${LLVM_BC_GENERATOR} to generate bitcode.")
        endif()

        if (NOT LLVM_AS_TOOL)
            find_program (LLVM_AS_TOOL NAMES "llvm-as"
                        PATHS "${LLVM_DIRECTORY}/bin" "${LLVM_DIRECTORY}/tools/llvm"
                        NO_CMAKE_PATH NO_DEFAULT_PATH NO_CMAKE_SYSTEM_PATH
                        NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_ENVIRONMENT_PATH)
        endif ()

        if (NOT LLVM_LINK_TOOL)
            find_program (LLVM_LINK_TOOL NAMES "llvm-link"
                    PATHS "${LLVM_DIRECTORY}/bin" "${LLVM_DIRECTORY}/tools/llvm"
                    NO_CMAKE_PATH NO_DEFAULT_PATH NO_CMAKE_SYSTEM_PATH
                    NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_ENVIRONMENT_PATH)
        endif ()


        # Fix specific problem I had on new Apple systems (e.g. Mavericks) with
        # LLVM/libc++ installed -- for some reason, LLVM 3.4 wasn't finding it,
        # so in that specific case, append another -I to point it in the right
        # direction.
        #if (APPLE AND ${LLVM_BC_GENERATOR} MATCHES ".*clang.*")
        #    exec_program ( "${LLVM_BC_GENERATOR}" ARGS --version OUTPUT_VARIABLE MY_CLANG_VERSION )
        #    string (REGEX REPLACE "clang version ([0-9][.][0-9]+).*" "\\1" MY_CLANG_VERSION "${MY_CLANG_VERSION}")
        #    if ((${MY_CLANG_VERSION} VERSION_GREATER "3.3")
        #          AND (EXISTS "/usr/lib/libc++.dylib")
        #          AND (EXISTS "/Library/Developer/CommandLineTools/usr/lib/c++/v1"))
        #        set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} "-I/Library/Developer/CommandLineTools/usr/lib/c++/v1")
        #    endif ()
        #endif ()

        list (TRANSFORM IMATH_INCLUDES PREPEND -I
            OUTPUT_VARIABLE ALL_IMATH_INCLUDES)
        list (TRANSFORM OPENEXR_INCLUDES PREPEND -I
            OUTPUT_VARIABLE ALL_OPENEXR_INCLUDES)
        list (TRANSFORM OpenImageIO_INCLUDES PREPEND -I
            OUTPUT_VARIABLE ALL_OpenImageIO_INCLUDES)


        # Command to turn the .cpp file into LLVM assembly language .s, into
        # LLVM bitcode .bc, then back into a C++ file with the bc embedded!
        add_custom_command ( OUTPUT ${src_bc}
        COMMAND ${LLVM_BC_GENERATOR}
            ${LLVM_COMPILE_FLAGS}
            "-I${CMAKE_CURRENT_SOURCE_DIR}"
            "-I${CMAKE_SOURCE_DIR}/src/include"
            "-I${CMAKE_BINARY_DIR}/include"
            ${ALL_OpenImageIO_INCLUDES}
            ${ALL_IMATH_INCLUDES}
            #"-isystem ${Boost_INCLUDE_DIRS}" #Does not pick up usr installed boost/thread/tss.hpp for oslexec_pvt.h
            "-I${Boost_INCLUDE_DIRS}"
            -DOSL_COMPILING_TO_BITCODE=1
            -Wno-deprecated-register
            # the following 2 warnings can be restored when all 3rd parties have fixed their export macros
            # (dllimport attribute is not supported when compiling for Cuda and triggers a ton of warnings)
            -Wno-ignored-attributes -Wno-unknown-attributes
            -O3 -fno-math-errno -S -emit-llvm ${extra_clang_args}
            -o ${src_asm} ${src}
        COMMAND ${LLVM_AS_TOOL} -f -o ${src_bc} ${src_asm}
        # Do NOT setup a MAIN_DEPENDENCY because only 1 may exist
        # and we may have the several outputs dependent on the same source
        DEPENDS ${src} ${exec_headers} ${PROJECT_PUBLIC_HEADERS}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
    endforeach ()


    if (VERBOSE)
        message ( STATUS "^^^^^^^^^^^^^^^^^^^^^^^^^^" )
        message ( STATUS "src_bc_list: ${src_bc_list} ")
        message ( STATUS "^^^^^^^^^^^^^^^^^^^^^^^^^^" )
    endif()

    # Link all of the individual LLVM bitcode files
    set ( linked_src_bc "${CMAKE_CURRENT_BINARY_DIR}/${output_name}.bc" )
    add_custom_command ( OUTPUT ${linked_src_bc}
        COMMAND ${LLVM_LINK_TOOL} -internalize ${src_bc_list} -o ${linked_src_bc}
        DEPENDS ${src_bc_list} ${exec_headers} ${PROJECT_PUBLIC_HEADERS} ${src_list}
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" )
    
     # Serialize the linked bitcode into a CPP file 
    set ( src_bc_cpp "${CMAKE_CURRENT_BINARY_DIR}/${output_name}.bc.cpp" )
    add_custom_command ( OUTPUT ${src_bc_cpp}
        COMMAND ${Python_EXECUTABLE} "${CMAKE_SOURCE_DIR}/src/build-scripts/serialize-bc.py"
            ${linked_src_bc} ${src_bc_cpp} ${output_name}
        DEPENDS "${CMAKE_SOURCE_DIR}/src/build-scripts/serialize-bc.py" ${linked_src_bc}
        ${exec_headers} ${PROJECT_PUBLIC_HEADERS}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

    # add generated .cpp with embedded bitcode to the list of sources
    set ( ${list_to_append_cpp} ${${list_to_append_cpp}} ${src_bc_cpp} PARENT_SCOPE )

endfunction ( )
