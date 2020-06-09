# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

set (local_lib oslexec)
set (lib_src
          shadingsys.cpp closure.cpp
          dictionary.cpp
          context.cpp instance.cpp
          loadshader.cpp master.cpp
          opcolor.cpp opmatrix.cpp opmessage.cpp
          opnoise.cpp
          opspline.cpp opstring.cpp optexture.cpp
          oslexec.cpp
          pointcloud.cpp rendservices.cpp
          constfold.cpp runtimeoptimize.cpp typespec.cpp
          lpexp.cpp lpeparse.cpp automata.cpp accum.cpp
          opclosure.cpp
          shadeimage.cpp
          backendllvm.cpp
          llvm_gen.cpp llvm_instance.cpp llvm_util.cpp
          ../liboslnoise/gabornoise.cpp
          ../liboslnoise/simplexnoise.cpp
    )

if (BUILD_SHARED_LIBS)
    # oslcomp and oslquery symbols used in oslexec
    list(APPEND lib_src
        ../liboslcomp/ast.cpp
        ../liboslcomp/codegen.cpp
        ../liboslcomp/oslcomp.cpp
        ../liboslcomp/symtab.cpp
        ../liboslcomp/typecheck.cpp
        ../liboslquery/oslquery.cpp
        )
endif ()

file (GLOB exec_headers "*.h")
file (GLOB compiler_headers "../liboslcomp/*.h")

FLEX_BISON ( osolex.l osogram.y oso lib_src exec_headers )
FLEX_BISON ( ../liboslcomp/osllex.l ../liboslcomp/oslgram.y osl lib_src compiler_headers )

set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS" )

macro ( LLVM_COMPILE llvm_src srclist )
    get_filename_component ( llvmsrc_we ${llvm_src} NAME_WE )
    set ( llvm_asm "${CMAKE_CURRENT_BINARY_DIR}/${llvmsrc_we}.s" )
    set ( llvm_bc "${CMAKE_CURRENT_BINARY_DIR}/${llvmsrc_we}.bc" )
    set ( llvm_bc_cpp "${CMAKE_CURRENT_BINARY_DIR}/${llvmsrc_we}.bc.cpp" )
    if (VERBOSE)
        message (STATUS "LLVM_COMPILE in=${llvm_src}")
        message (STATUS "LLVM_COMPILE asm=${llvm_asm}")
        message (STATUS "LLVM_COMPILE bc=${llvm_bc}")
        message (STATUS "LLVM_COMPILE cpp=${llvm_bc_cpp}")
    endif ()
    set ( ${srclist} ${${srclist}} ${llvm_bc_cpp} )
    get_property (CURRENT_DEFINITIONS DIRECTORY PROPERTY COMPILE_DEFINITIONS)
    if (VERBOSE)
        message (STATUS "Current #defines are ${CURRENT_DEFINITIONS}")
    endif ()
    foreach (def ${CURRENT_DEFINITIONS})
        set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} "-D${def}")
    endforeach()
    set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} ${SIMD_COMPILE_FLAGS} ${CSTD_FLAGS} ${TOOLCHAIN_FLAGS})

    # Figure out what program we will use to make the bitcode.
    if (NOT LLVM_BC_GENERATOR)
        find_program (LLVM_BC_GENERATOR NAMES "clang++"
                      PATHS "${LLVM_DIRECTORY}/bin"
                      NO_CMAKE_PATH NO_DEFAULT_PATH NO_CMAKE_SYSTEM_PATH
                      NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_ENVIRONMENT_PATH)
    endif ()
    # If that didn't work, look anywhere
    if (NOT LLVM_BC_GENERATOR)
        # Wasn't in their build, look anywhere
        find_program (LLVM_BC_GENERATOR NAMES clang++ llvm-g++)
    endif ()

    if (NOT LLVM_BC_GENERATOR)
        message (FATAL_ERROR "You must have a valid llvm bitcode generator (clang++) somewhere.")
    endif ()
    if (VERBOSE)
        message (STATUS "Using LLVM_BC_GENERATOR ${LLVM_BC_GENERATOR} to generate bitcode.")
    endif()

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

    # Command to turn the .cpp file into LLVM assembly language .s, into
    # LLVM bitcode .bc, then back into a C++ file with the bc embedded!
    add_custom_command ( OUTPUT ${llvm_bc_cpp}
      COMMAND ${LLVM_BC_GENERATOR}
          ${LLVM_COMPILE_FLAGS}
          "-I${CMAKE_CURRENT_SOURCE_DIR}"
          "-I${CMAKE_SOURCE_DIR}/src/include"
          "-I${CMAKE_BINARY_DIR}/include"
          "-I${OPENIMAGEIO_INCLUDES}"
          "-I${ILMBASE_INCLUDES}"
          "-isystem ${Boost_INCLUDE_DIRS}"
          -DOSL_COMPILING_TO_BITCODE=1
          -Wno-deprecated-register
          -O3 -fno-math-errno -S -emit-llvm -o ${llvm_asm} ${llvm_src}
      COMMAND "${LLVM_DIRECTORY}/bin/llvm-as" -f -o ${llvm_bc} ${llvm_asm}
      COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/serialize-bc.py" ${llvm_bc} ${llvm_bc_cpp} "osl_llvm_compiled_ops"
      MAIN_DEPENDENCY ${llvm_src}
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/serialize-bc.py"
              ${exec_headers} ${PROJECT_PUBLIC_HEADERS}
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
endmacro ( )


macro ( CUDA_SHADEOPS_COMPILE srclist )
    # Add all of the "shadeops" sources that need to be compiled to LLVM bitcode for CUDA
    set ( shadeops_srcs
        llvm_ops.cpp
        opnoise.cpp
        opspline.cpp
        opcolor.cpp
        opmatrix.cpp
        ../liboslnoise/gabornoise.cpp
        ../liboslnoise/simplexnoise.cpp
        )

    set ( shadeops_bc_cuda_cpp "${CMAKE_CURRENT_BINARY_DIR}/shadeops_cuda.bc.cpp" )
    set ( linked_shadeops_bc "${CMAKE_CURRENT_BINARY_DIR}/linked_shadeops.bc" )

    list ( APPEND ${srclist} ${shadeops_bc_cuda_cpp} )

    foreach ( shadeops_src ${shadeops_srcs} )
        MAKE_CUDA_BITCODE ( ${shadeops_src} "_cuda" shadeops_bc "" )
        list ( APPEND shadeops_bc_list ${shadeops_bc} )
    endforeach ()

    # Link all of the individual LLVM bitcode files
    add_custom_command ( OUTPUT ${linked_shadeops_bc}
        COMMAND "${LLVM_DIRECTORY}/bin/llvm-link" -internalize ${shadeops_bc_list} -o ${linked_shadeops_bc}
        DEPENDS ${shadeops_bc_list} ${exec_headers} ${PROJECT_PUBLIC_HEADERS} ${shadeops_srcs}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

    # Serialze the linked bitcode into a CPP file and add it to the list of liboslexec soures
    add_custom_command ( OUTPUT ${shadeops_bc_cuda_cpp}
        COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/serialize-bc.py"
            ${linked_shadeops_bc} ${shadeops_bc_cuda_cpp} "osl_llvm_compiled_ops_cuda"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/serialize-bc.py" ${linked_shadeops_bc}
        ${exec_headers} ${PROJECT_PUBLIC_HEADERS}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
endmacro ( )

if (USE_LLVM_BITCODE)
    LLVM_COMPILE ( llvm_ops.cpp lib_src )

    # Optionally repeat the bitcode compilation with CUDA-specific options
    if (CUDA_FOUND)
        add_definitions (-DOSL_LLVM_CUDA_BITCODE)
        CUDA_SHADEOPS_COMPILE ( lib_src )
    endif ()
else ()
    # With MSVC/Mingw, we don't compile llvm_ops.cpp to LLVM bitcode, due
    # to clang being unable to compile MSVC C++ header files at this time.
    # Instead it is part of the regular build process.
    add_definitions (-DOSL_LLVM_NO_BITCODE)
    set (lib_src ${lib_src} llvm_ops.cpp)
endif ()

add_library (${local_lib} ${lib_src})
target_include_directories (${local_lib}
    PUBLIC
        ${CMAKE_INSTALL_FULL_INCLUDEDIR}
        ${OPENIMAGEIO_INCLUDES}
        ${ILMBASE_INCLUDES}
    PRIVATE
        "${CMAKE_SOURCE_DIR}/src/liboslcomp"
        ${PUGIXML_INCLUDES}
    )
target_compile_definitions (${local_lib} PRIVATE OSL_EXPORTS)
if (partio_FOUND)
    target_link_libraries(${local_lib} PRIVATE partio::partio)
    target_compile_definitions (${local_lib} PRIVATE USE_PARTIO=1)
endif ()

# Propagate our library's RTTI options to downstream clients of this library
# This is very important for proper linkage with LLVM.
target_compile_options(${local_lib} PUBLIC ${${PROJECT_NAME}_RTTI_OPTIONS})

target_link_libraries (${local_lib}
    PUBLIC
        ${OPENIMAGEIO_LIBRARIES} ${ILMBASE_LIBRARIES}
    PRIVATE
        ${PUGIXML_LIBRARIES}
        ${ZLIB_LIBRARIES}
        ${Boost_LIBRARIES} ${CMAKE_DL_LIBS}
        ${CLANG_LIBRARIES}
        ${LLVM_LIBRARIES} ${LLVM_LDFLAGS} ${LLVM_SYSTEM_LIBRARIES}
    )

set_target_properties (${local_lib}
        PROPERTIES
        VERSION     ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
        SOVERSION ${SOVERSION}
        OUTPUT_NAME ${local_lib}${OSL_LIBNAME_SUFFIX}
    )

if (VISIBILITY_COMMAND)
    set_property (TARGET ${local_lib}
                  APPEND PROPERTY LINK_FLAGS ${VISIBILITY_MAP_COMMAND})
endif ()

# add_dependencies (${local_lib} "${CMAKE_SOURCE_DIR}/src/build-scripts/hidesymbols.map")

install_targets (${local_lib})

# Unit tests
if (OSL_BUILD_TESTS)
    add_executable (accum_test accum_test.cpp)
    target_link_libraries (accum_test PRIVATE oslexec ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
    set_target_properties (accum_test PROPERTIES FOLDER "Unit Tests")
    add_test (unit_accum ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/accum_test)

    add_executable (dual_test dual_test.cpp)
    target_link_libraries (dual_test PRIVATE ${OPENIMAGEIO_LIBRARIES} ${ILMBASE_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
    set_target_properties (dual_test PROPERTIES FOLDER "Unit Tests")
    add_test (unit_dual ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/dual_test)

    add_executable (llvmutil_test llvmutil_test.cpp)
    target_link_libraries (llvmutil_test PRIVATE oslexec ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
    set_target_properties (llvmutil_test PROPERTIES FOLDER "Unit Tests")
    add_test (unit_llvmutil ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llvmutil_test)
endif ()
