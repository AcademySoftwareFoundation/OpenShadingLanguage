# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# TODO: Use CMAKE_CURRENT_FUNCTION_LIST_DIR in cmake-3.17
set(_THIS_MODULE_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function ( EMBED_LLVM_BITCODE_IN_CPP src_list suffix output_name list_to_append_cpp extra_clang_args include_dirs)

    message (VERBOSE "EMBED_LLVM_BITCODE_IN_CPP src_list=${src_list}")

    foreach ( src ${src_list} )
        get_filename_component ( src_we ${src} NAME_WE )
        set ( src_asm "${CMAKE_CURRENT_BINARY_DIR}/${src_we}${suffix}.s" )
        set ( src_bc "${CMAKE_CURRENT_BINARY_DIR}/${src_we}${suffix}.bc" )
        message (VERBOSE "EMBED_LLVM_BITCODE_IN_CPP in=${src}")
        message (VERBOSE "EMBED_LLVM_BITCODE_IN_CPP asm=${src_asm}")
        message (VERBOSE "EMBED_LLVM_BITCODE_IN_CPP bc=${src_bc}")
        list ( APPEND src_bc_list ${src_bc} )

        get_property (CURRENT_DEFINITIONS DIRECTORY PROPERTY COMPILE_DEFINITIONS)
        message (VERBOSE "Current #defines are ${CURRENT_DEFINITIONS}")
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
        if (LLVM_BC_GENERATOR)
            execute_process ( COMMAND ${LLVM_BC_GENERATOR} -dumpversion
                              OUTPUT_VARIABLE LLVM_BC_GENERATOR_VERSION
                              OUTPUT_STRIP_TRAILING_WHITESPACE )
        endif ()
        message (VERBOSE "Using LLVM_BC_GENERATOR ${LLVM_BC_GENERATOR} v${LLVM_BC_GENERATOR_VERSION} to generate bitcode.")

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

        if (APPLE)
            if (CMAKE_OSX_SYSROOT)
                list (APPEND LLVM_COMPILE_FLAGS -isysroot ${CMAKE_OSX_SYSROOT})
            endif ()
            if (CMAKE_OSX_DEPLOYMENT_TARGET)
                list (APPEND LLVM_COMPILE_FLAGS -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET})
            endif ()
            if (CMAKE_OSX_ARCHITECTURES)
                list (APPEND LLVM_COMPILE_FLAGS -arch ${CMAKE_OSX_ARCHITECTURES})
            endif ()
        endif ()

        list (TRANSFORM include_dirs PREPEND -I
            OUTPUT_VARIABLE ALL_INCLUDE_DIRS)

        # Command to turn the .cpp file into LLVM assembly language .s, into
        # LLVM bitcode .bc, then back into a C++ file with the bc embedded!
        add_custom_command ( OUTPUT ${src_bc}
        COMMAND ${LLVM_BC_GENERATOR}
            ${LLVM_COMPILE_FLAGS}
            ${ALL_INCLUDE_DIRS}
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

    message (VERBOSE "^^^^^^^^^^^^^^^^^^^^^^^^^^" )
    message (VERBOSE "src_bc_list: ${src_bc_list} ")
    message (VERBOSE "^^^^^^^^^^^^^^^^^^^^^^^^^^" )

    # Link all of the individual LLVM bitcode files
    set ( linked_src_bc "${CMAKE_CURRENT_BINARY_DIR}/${output_name}.bc" )
    add_custom_command ( OUTPUT ${linked_src_bc}
        COMMAND ${LLVM_LINK_TOOL} -internalize ${src_bc_list} -o ${linked_src_bc}
        DEPENDS ${src_bc_list} ${exec_headers} ${PROJECT_PUBLIC_HEADERS} ${src_list}
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" )
    
     # Serialize the linked bitcode into a CPP file 
    set ( src_bc_cpp "${CMAKE_CURRENT_BINARY_DIR}/${output_name}.bc.cpp" )
    add_custom_command ( OUTPUT ${src_bc_cpp}
        COMMAND ${Python3_EXECUTABLE} "${_THIS_MODULE_BASE_DIR}/../build-scripts/serialize-bc.py"
            ${linked_src_bc} ${src_bc_cpp} ${output_name}
        DEPENDS "${_THIS_MODULE_BASE_DIR}/../build-scripts/serialize-bc.py" ${linked_src_bc}
        ${exec_headers} ${PROJECT_PUBLIC_HEADERS}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

    # add generated .cpp with embedded bitcode to the list of sources
    set ( ${list_to_append_cpp} ${${list_to_append_cpp}} ${src_bc_cpp} PARENT_SCOPE )

endfunction ( )
