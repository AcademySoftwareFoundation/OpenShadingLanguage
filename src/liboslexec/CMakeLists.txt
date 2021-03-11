# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# NOTE: for any additions/deletions from liboslexec_target_srcs,
# please update wide_target_combine_text_and_rodata.ld
set ( liboslexec_target_srcs
    wide/wide_opalgebraic    
    )

set ( liboslexec_override_limits
    )
    
# Strategy is to make a copy of each cpp in liboslexec_target_srcs for 
# each target batch width and ISA combination, applying compiler flags to define 
# -D__OSL_WIDTH=[4|8|16]
# -D__OSL_TARGET_ISA=[AVX512|AVX2|AVX|SSE4_2]
# and compiler flags to choose correct target ISA for your compiler
# You may then add/remove the desired  ${TARGET_SOURCES_B[4|8|16]_[AVX512|AVX2|AVX|SSE4_2]}
# to your add_library call 
foreach(target_src ${liboslexec_target_srcs})
    
    set(DST_B16_AVX512 "${CMAKE_CURRENT_BINARY_DIR}/${target_src}_b16_AVX512.cpp")
    set(DST_B16_AVX512_NOFMA "${CMAKE_CURRENT_BINARY_DIR}/${target_src}_b16_AVX512_noFMA.cpp")
    set(DST_B8_AVX512 "${CMAKE_CURRENT_BINARY_DIR}/${target_src}_b8_AVX512.cpp")
    set(DST_B8_AVX512_NOFMA "${CMAKE_CURRENT_BINARY_DIR}/${target_src}_b8_AVX512_noFMA.cpp")
    set(DST_B8_AVX2 "${CMAKE_CURRENT_BINARY_DIR}/${target_src}_b8_AVX2.cpp")
    set(DST_B8_AVX2_NOFMA "${CMAKE_CURRENT_BINARY_DIR}/${target_src}_b8_AVX2_noFMA.cpp")
    set(DST_B8_AVX "${CMAKE_CURRENT_BINARY_DIR}/${target_src}_b8_AVX.cpp")
    
    set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/${target_src}.cpp")
    
    CONFIGURE_FILE("${SRC}" ${DST_B16_AVX512} COPYONLY)
    CONFIGURE_FILE("${SRC}" ${DST_B16_AVX512_NOFMA} COPYONLY)
    CONFIGURE_FILE(${SRC} ${DST_B8_AVX512} COPYONLY)
    CONFIGURE_FILE(${SRC} ${DST_B8_AVX512_NOFMA} COPYONLY)
    CONFIGURE_FILE(${SRC} ${DST_B8_AVX2} COPYONLY)
    CONFIGURE_FILE(${SRC} ${DST_B8_AVX2_NOFMA} COPYONLY)
    CONFIGURE_FILE(${SRC} ${DST_B8_AVX} COPYONLY)
    
    list(APPEND TARGET_SOURCES_B16_AVX512 "${DST_B16_AVX512}")
    list(APPEND TARGET_SOURCES_B16_AVX512_NOFMA "${DST_B16_AVX512_NOFMA}")
    list(APPEND TARGET_SOURCES_B8_AVX512 "${DST_B8_AVX512}")
    list(APPEND TARGET_SOURCES_B8_AVX512_NOFMA "${DST_B8_AVX512_NOFMA}")
    list(APPEND TARGET_SOURCES_B8_AVX2 "${DST_B8_AVX2}")
    list(APPEND TARGET_SOURCES_B8_AVX2_NOFMA "${DST_B8_AVX2_NOFMA}")
    list(APPEND TARGET_SOURCES_B8_AVX "${DST_B8_AVX}")
    
    set_property(SOURCE "${DST_B16_AVX512}" "${DST_B16_AVX512_NOFMA}" "${DST_B8_AVX512}" "${DST_B8_AVX512_NOFMA}" "${DST_B8_AVX2}" "${DST_B8_AVX2_NOFMA}" "${DST_B8_AVX}" "${DST_B4_SSE4_2}" 
        APPEND PROPERTY COMPILE_OPTIONS 
        "-I${CMAKE_CURRENT_SOURCE_DIR}/wide"
        "-I${CMAKE_CURRENT_SOURCE_DIR}/../liboslnoise/wide"
        )
    
    set_property(SOURCE "${DST_B16_AVX512}" "${DST_B16_AVX512_NOFMA}"
        APPEND PROPERTY COMPILE_DEFINITIONS "__OSL_WIDTH=16")
    set_property(SOURCE "${DST_B8_AVX512}" "${DST_B8_AVX512_NOFMA}" "${DST_B8_AVX2}" "${DST_B8_AVX2_NOFMA}" "${DST_B8_AVX}"
        APPEND PROPERTY COMPILE_DEFINITIONS "__OSL_WIDTH=8")
    
    set_property(SOURCE "${DST_B16_AVX512}" "${DST_B8_AVX512}"
        APPEND PROPERTY COMPILE_DEFINITIONS "__OSL_TARGET_ISA=AVX512")
    set_property(SOURCE "${DST_B16_AVX512_NOFMA}" "${DST_B8_AVX512_NOFMA}"
        APPEND PROPERTY COMPILE_DEFINITIONS "__OSL_TARGET_ISA=AVX512_noFMA")
    set_property(SOURCE "${DST_B8_AVX2}" 
        APPEND PROPERTY COMPILE_DEFINITIONS "__OSL_TARGET_ISA=AVX2")
    set_property(SOURCE "${DST_B8_AVX2_NOFMA}" 
        APPEND PROPERTY COMPILE_DEFINITIONS "__OSL_TARGET_ISA=AVX2_noFMA")
    set_property(SOURCE "${DST_B8_AVX}" 
        APPEND PROPERTY COMPILE_DEFINITIONS "__OSL_TARGET_ISA=AVX")
    
    if (CMAKE_COMPILER_IS_INTEL)
        if (MSVC)
            set_property(SOURCE "${DST_B16_AVX512}" "${DST_B16_AVX512_NOFMA}" 
                APPEND PROPERTY COMPILE_OPTIONS "/QxCORE-AVX512" "/Qopt-zmm-usage:high")
            set_property(SOURCE "${DST_B8_AVX512}" "${DST_B8_AVX512_NOFMA}" 
                APPEND PROPERTY COMPILE_OPTIONS "/QxCORE-AVX512" "/Qopt-zmm-usage:low")
            set_property(SOURCE "${DST_B8_AVX2}" "${DST_B8_AVX2_NOFMA}" 
                APPEND PROPERTY COMPILE_OPTIONS "/QxCORE-AVX2")
            set_property(SOURCE "${DST_B8_AVX}" 
                APPEND PROPERTY COMPILE_OPTIONS "/QxAVX")
            
            set_property(SOURCE "${DST_B16_AVX512_NOFMA}" "${DST_B8_AVX512_NOFMA}" "${DST_B8_AVX2_NOFMA}"
                APPEND PROPERTY COMPILE_OPTIONS "/Qfma-")
            
            if (${target_src} IN_LIST liboslexec_override_limits)
                set_property(SOURCE "${DST_B16_AVX512}" "${DST_B16_AVX512_NOFMA}" "${DST_B8_AVX512}" "${DST_B8_AVX512_NOFMA}"
                "${DST_B8_AVX2}" "${DST_B8_AVX2_NOFMA}" "${DST_B8_AVX}"
                  APPEND PROPERTY COMPILE_OPTIONS "/Qoverride-limits")
            endif()
            set_property(SOURCE "${DST_B16_AVX512}" "${DST_B16_AVX512_NOFMA}" "${DST_B8_AVX512}" "${DST_B8_AVX512_NOFMA}"
            "${DST_B8_AVX2}" "${DST_B8_AVX2_NOFMA}" "${DST_B8_AVX}"
              APPEND PROPERTY COMPILE_OPTIONS "/Qprec-div" "/Qqopt-report:5")
        else ()
            set_property(SOURCE "${DST_B16_AVX512}" "${DST_B16_AVX512_NOFMA}" 
                APPEND PROPERTY COMPILE_OPTIONS "-xCORE-AVX512" "-qopt-zmm-usage=high")
            set_property(SOURCE "${DST_B8_AVX512}" "${DST_B8_AVX512_NOFMA}" 
                APPEND PROPERTY COMPILE_OPTIONS "-xCORE-AVX512" "-qopt-zmm-usage=low")
            set_property(SOURCE "${DST_B8_AVX2}" "${DST_B8_AVX2_NOFMA}" 
                APPEND PROPERTY COMPILE_OPTIONS "-xCORE-AVX2")
            set_property(SOURCE "${DST_B8_AVX}" 
                APPEND PROPERTY COMPILE_OPTIONS "-xAVX")
            
            set_property(SOURCE "${DST_B16_AVX512_NOFMA}" "${DST_B8_AVX512_NOFMA}" "${DST_B8_AVX2_NOFMA}"
                APPEND PROPERTY COMPILE_OPTIONS "-no-fma")
            
            if (${target_src} IN_LIST liboslexec_override_limits)
                set_property(SOURCE "${DST_B16_AVX512}" "${DST_B16_AVX512_NOFMA}" "${DST_B8_AVX512}" "${DST_B8_AVX512_NOFMA}"
                "${DST_B8_AVX2}" "${DST_B8_AVX2_NOFMA}" "${DST_B8_AVX}"
                  APPEND PROPERTY COMPILE_OPTIONS "-qoverride-limits")
            endif()
            set_property(SOURCE "${DST_B16_AVX512}" "${DST_B16_AVX512_NOFMA}" "${DST_B8_AVX512}" "${DST_B8_AVX512_NOFMA}"
            "${DST_B8_AVX2}" "${DST_B8_AVX2_NOFMA}" "${DST_B8_AVX}"
              APPEND PROPERTY COMPILE_OPTIONS "-prec-div" "-qopt-report=5")
        endif ()
    endif ()
    
    if (CMAKE_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_APPLECLANG)
        # REQUIRES CLANG 7+ to use sucessfully vectorize with OpenMP simd
        set_property(SOURCE "${DST_B16_AVX512}" "${DST_B16_AVX512_NOFMA}" "${DST_B8_AVX512}" "${DST_B8_AVX512_NOFMA}"
            APPEND PROPERTY COMPILE_OPTIONS "-march=skylake-avx512")
        if (CLANG_VERSION_STRING VERSION_GREATER 7.0)
            set_property(SOURCE "${DST_B16_AVX512}" "${DST_B16_AVX512_NOFMA}" 
                APPEND PROPERTY COMPILE_OPTIONS "-mprefer-vector-width=512")
            set_property(SOURCE "${DST_B8_AVX512}" "${DST_B8_AVX512_NOFMA}" 
                APPEND PROPERTY COMPILE_OPTIONS "-mprefer-vector-width=256")
        endif ()
        set_property(SOURCE "${DST_B8_AVX2}" "${DST_B8_AVX2_NOFMA}" 
            APPEND PROPERTY COMPILE_OPTIONS "-march=core-avx2")
        set_property(SOURCE "${DST_B8_AVX}" 
            APPEND PROPERTY COMPILE_OPTIONS "-march=corei7-avx")
        
        set_property(SOURCE "${DST_B16_AVX512_NOFMA}" "${DST_B8_AVX512_NOFMA}" "${DST_B8_AVX2_NOFMA}"
            APPEND PROPERTY COMPILE_OPTIONS "-mno-fma")        
            
        # large SIMD function loops will exceed llvm's -inline-threshold default of 225.
        # remark: loop not vectorized: call instruction cannot be vectorized [-Rpass-analysis]     
        # choose to increase that limit via compiler flags vs. 
        # workaround with __attribute__((flatten))     
        set_property(SOURCE "${DST_B16_AVX512}" "${DST_B16_AVX512_NOFMA}" "${DST_B8_AVX512}" "${DST_B8_AVX512_NOFMA}"
        "${DST_B8_AVX2}" "${DST_B8_AVX2_NOFMA}" "${DST_B8_AVX}"
          APPEND PROPERTY COMPILE_OPTIONS "-mllvm" "-inline-threshold=100000")
            
        # For loops with small loop bodies, clang was unrolling the loop before
        # #pragma omp simd
        # was processed.  Thus there was no loop left for the pragma to apply to
        # To work around this, for these wide SIMD libraries, we will disable
        # clang's loop unrolling.  We really don't want to do this and wish to remove
        # it as soon as clang addresses this issue.
        set_property(SOURCE "${DST_B16_AVX512}" "${DST_B16_AVX512_NOFMA}" "${DST_B8_AVX512}" "${DST_B8_AVX512_NOFMA}"
        "${DST_B8_AVX2}" "${DST_B8_AVX2_NOFMA}" "${DST_B8_AVX}"
          APPEND PROPERTY COMPILE_OPTIONS "-fno-unroll-loops")
    endif ()

endforeach(target_src)

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
          batched_rendservices.cpp
          constfold.cpp runtimeoptimize.cpp typespec.cpp
          lpexp.cpp lpeparse.cpp automata.cpp accum.cpp
          opclosure.cpp
          shadeimage.cpp
          backendllvm.cpp
          llvm_gen.cpp llvm_instance.cpp llvm_util.cpp
          batched_analysis.cpp
          batched_backendllvm.cpp
          batched_llvm_gen.cpp
          batched_llvm_instance.cpp
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

    list (TRANSFORM IMATH_INCLUDES PREPEND -I
          OUTPUT_VARIABLE ALL_IMATH_INCLUDES)

    # Command to turn the .cpp file into LLVM assembly language .s, into
    # LLVM bitcode .bc, then back into a C++ file with the bc embedded!
    add_custom_command ( OUTPUT ${llvm_bc_cpp}
      COMMAND ${LLVM_BC_GENERATOR}
          ${LLVM_COMPILE_FLAGS}
          "-I${CMAKE_CURRENT_SOURCE_DIR}"
          "-I${CMAKE_SOURCE_DIR}/src/include"
          "-I${CMAKE_BINARY_DIR}/include"
          "-I${OpenImageIO_INCLUDES}"
          ${ALL_IMATH_INCLUDES}
          "-isystem ${Boost_INCLUDE_DIRS}"
          -DOSL_COMPILING_TO_BITCODE=1
          -Wno-deprecated-register
          # the following 2 warnings can be restored when all 3rd parties have fixed their export macros
          # (dllimport attribute is not supported when compiling for Cuda and triggers a ton of warnings)
          -Wno-ignored-attributes -Wno-unknown-attributes
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

    # Serialize the linked bitcode into a CPP file and add it to the list of liboslexec soures
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

# Always build target specific libraries for batched execution as shared
# For each target specific shared library being built, we need
# we need to let a couple source files know so they can build
# function mappings for those targets
set_property(SOURCE "batched_llvm_instance.cpp"  "shadingsys.cpp" 
    APPEND PROPERTY COMPILE_DEFINITIONS 
        "__OSL_SUPPORTS_B16_AVX512"
        "__OSL_SUPPORTS_B8_AVX512"
        "__OSL_SUPPORTS_B8_AVX2"
        "__OSL_SUPPORTS_B8_AVX"
        #"__OSL_SUPPORTS_B16_AVX512_NOFMA"
        #"__OSL_SUPPORTS_B8_AVX512_NOFMA"
        #"__OSL_SUPPORTS_B8_AVX2_NOFMA"
    )
  

set (_b16_AVX512_oslexec_lib _b16_AVX512_oslexec)
set (_b8_AVX512_oslexec_lib _b8_AVX512_oslexec)
set (_b8_AVX2_oslexec_lib _b8_AVX2_oslexec)
set (_b8_AVX_oslexec_lib _b8_AVX_oslexec)
set (_b16_AVX512_noFMA_oslexec_lib _b16_AVX512_noFMA_oslexec)
set (_b8_AVX512_noFMA_oslexec_lib _b8_AVX512_noFMA_oslexec)
set (_b8_AVX2_noFMA_oslexec_lib _b8_AVX2_noFMA_oslexec)

add_library ( ${_b16_AVX512_oslexec_lib} SHARED "${TARGET_SOURCES_B16_AVX512}" )
add_library ( ${_b8_AVX512_oslexec_lib} SHARED "${TARGET_SOURCES_B8_AVX512}" )
add_library ( ${_b8_AVX2_oslexec_lib} SHARED "${TARGET_SOURCES_B8_AVX2}" )
add_library ( ${_b8_AVX_oslexec_lib} SHARED "${TARGET_SOURCES_B8_AVX}" )
#add_library ( ${_b16_AVX512_noFMA_oslexec_lib} SHARED "${TARGET_SOURCES_B16_AVX512_NOFMA}" )
#add_library ( ${_b8_AVX512_noFMA_oslexec_lib} SHARED "${TARGET_SOURCES_B8_AVX512_NOFMA}" )
#add_library ( ${_b8_AVX2_noFMA_oslexec_lib} SHARED "${TARGET_SOURCES_B8_AVX2_NOFMA}" )

set ( liboslexec_target_libs
    ${_b16_AVX512_oslexec_lib}
    ${_b8_AVX512_oslexec_lib}  
    ${_b8_AVX2_oslexec_lib} 
    ${_b8_AVX_oslexec_lib}
#    ${_b16_AVX512_noFMA_oslexec_lib}
#    ${_b8_AVX512_noFMA_oslexec_lib}
#    ${_b8_AVX2_noFMA_oslexec_lib}
    )

foreach(target_lib ${liboslexec_target_libs})
    target_include_directories (${target_lib}
        PUBLIC
            ${CMAKE_INSTALL_FULL_INCLUDEDIR}
            ${ILMBASE_INCLUDES}
        )
    target_compile_definitions (${target_lib}
        PRIVATE
            OSL_EXPORTS
            CUDA_TARGET_ARCH="${CUDA_TARGET_ARCH}"
            OSL_CUDA_VERSION="${CUDA_VERSION}"
            OSL_OPTIX_VERSION="${OPTIX_VERSION}"
        )
    target_link_libraries (${target_lib}
        PUBLIC
            OpenImageIO::OpenImageIO
            # For OpenEXR/Imath 3.x:
            $<$<TARGET_EXISTS:Imath::Imath>:Imath::Imath>
            $<$<TARGET_EXISTS:Imath::Half>:Imath::Half>
            # For OpenEXR >= 2.4/2.5 with reliable exported targets
            $<$<TARGET_EXISTS:IlmBase::Imath>:IlmBase::Imath>
            $<$<TARGET_EXISTS:IlmBase::Half>:IlmBase::Half>
            $<$<TARGET_EXISTS:IlmBase::IlmThread>:IlmBase::IlmThread>
            $<$<TARGET_EXISTS:IlmBase::Iex>:IlmBase::Iex>
            # For OpenEXR <= 2.3:
            ${ILMBASE_LIBRARIES}
        PRIVATE
            pugixml::pugixml
            ${ZLIB_LIBRARIES}
            ${Boost_LIBRARIES} ${CMAKE_DL_LIBS}
            ${CLANG_LIBRARIES}
            ${LLVM_LIBRARIES} ${LLVM_LDFLAGS} ${LLVM_SYSTEM_LIBRARIES}
        )
endforeach(target_lib)

add_library (${local_lib} ${lib_src})
target_include_directories (${local_lib}
    PUBLIC
        ${CMAKE_INSTALL_FULL_INCLUDEDIR}
        ${ILMBASE_INCLUDES}
    PRIVATE
        "${CMAKE_SOURCE_DIR}/src/liboslcomp"
    )
target_compile_definitions (${local_lib}
    PRIVATE
        OSL_EXPORTS
        CUDA_TARGET_ARCH="${CUDA_TARGET_ARCH}"
        OSL_CUDA_VERSION="${CUDA_VERSION}"
        OSL_OPTIX_VERSION="${OPTIX_VERSION}"
    )
if (partio_FOUND)
    target_link_libraries(${local_lib} PRIVATE partio::partio)
    target_compile_definitions (${local_lib} PRIVATE USE_PARTIO=1)
endif ()

# Disable RTTI from just the one module where we have classes that inherit
# from LLVM classes.
if (MSVC)
    set_source_files_properties (llvm_util.cpp
                                 PROPERTIES COMPILE_FLAGS "/GR-")
else ()
    set_source_files_properties (llvm_util.cpp
                                 PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

# link with (system) library to prevent missing symbols inside clangDriver.lib
if (MSVC)
    target_link_libraries (${local_lib} PRIVATE "Version.lib")
endif()

target_link_libraries (${local_lib}
    PUBLIC
        OpenImageIO::OpenImageIO
        # For OpenEXR/Imath 3.x:
        $<$<TARGET_EXISTS:Imath::Imath>:Imath::Imath>
        $<$<TARGET_EXISTS:Imath::Half>:Imath::Half>
        # For OpenEXR >= 2.4/2.5 with reliable exported targets
        $<$<TARGET_EXISTS:IlmBase::Imath>:IlmBase::Imath>
        $<$<TARGET_EXISTS:IlmBase::Half>:IlmBase::Half>
        $<$<TARGET_EXISTS:IlmBase::IlmThread>:IlmBase::IlmThread>
        $<$<TARGET_EXISTS:IlmBase::Iex>:IlmBase::Iex>
        # For OpenEXR <= 2.3:
        ${ILMBASE_LIBRARIES}
    PRIVATE
        pugixml::pugixml
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

if (VISIBILITY_MAP_COMMAND)
    set_property (TARGET ${local_lib}
                  APPEND PROPERTY LINK_FLAGS ${VISIBILITY_MAP_COMMAND})
endif ()

# add_dependencies (${local_lib} "${CMAKE_SOURCE_DIR}/src/build-scripts/hidesymbols.map")

install_targets ( _b16_AVX512_oslexec )
install_targets ( _b8_AVX512_oslexec )
install_targets ( _b8_AVX2_oslexec )
install_targets ( _b8_AVX_oslexec )
#install_targets ( _b16_AVX512_noFMA_oslexec )
#install_targets ( _b8_AVX512_noFMA_oslexec )
#install_targets ( _b8_AVX2_noFMA_oslexec )

install_targets (${local_lib})

# Unit tests
if (OSL_BUILD_TESTS)
    add_executable (accum_test accum_test.cpp)
    target_link_libraries (accum_test PRIVATE oslexec ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
    set_target_properties (accum_test PROPERTIES FOLDER "Unit Tests")
    add_test (unit_accum ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/accum_test)

    add_executable (dual_test dual_test.cpp)
    target_link_libraries (dual_test PRIVATE OpenImageIO::OpenImageIO ${ILMBASE_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
    set_target_properties (dual_test PROPERTIES FOLDER "Unit Tests")
    add_test (unit_dual ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/dual_test)

    add_executable (llvmutil_test llvmutil_test.cpp)
    target_link_libraries (llvmutil_test PRIVATE oslexec ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
    set_target_properties (llvmutil_test PROPERTIES FOLDER "Unit Tests")
    add_test (unit_llvmutil ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llvmutil_test)
endif ()
