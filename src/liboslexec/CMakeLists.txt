# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# NOTE: for any additions/deletions from liboslexec_target_srcs,
# please update wide_target_combine_text_and_rodata.ld
set ( liboslexec_target_srcs
    wide/wide_opalgebraic
    wide/wide_opmatrix
    wide/wide_opnoise
    wide/wide_opnoise_cell
    wide/wide_opnoise_gabor_impl
    wide/wide_opnoise_generic_impl
    wide/wide_opnoise_hash
    wide/wide_opnoise_null
    wide/wide_opnoise_null_deriv
    wide/wide_opnoise_periodic_cell
    wide/wide_opnoise_periodic_gabor_impl
    wide/wide_opnoise_periodic_generic_impl
    wide/wide_opnoise_periodic_hash
    wide/wide_opnoise_periodic_perlin_float
    wide/wide_opnoise_periodic_perlin_deriv_float
    wide/wide_opnoise_periodic_perlin_Vec3
    wide/wide_opnoise_periodic_perlin_deriv_Vec3
    wide/wide_opnoise_periodic_uperlin_float
    wide/wide_opnoise_periodic_uperlin_deriv_float
    wide/wide_opnoise_periodic_uperlin_Vec3
    wide/wide_opnoise_periodic_uperlin_deriv_Vec3
    wide/wide_opnoise_perlin_float
    wide/wide_opnoise_perlin_deriv_float
    wide/wide_opnoise_perlin_Vec3
    wide/wide_opnoise_perlin_deriv_Vec3
    wide/wide_opnoise_simplex_float
    wide/wide_opnoise_simplex_deriv_float
    wide/wide_opnoise_simplex_Vec3
    wide/wide_opnoise_simplex_deriv_Vec3
    wide/wide_opnoise_unull
    wide/wide_opnoise_unull_deriv
    wide/wide_opnoise_uperlin_float
    wide/wide_opnoise_uperlin_deriv_float
    wide/wide_opnoise_uperlin_Vec3
    wide/wide_opnoise_uperlin_deriv_Vec3
    wide/wide_opnoise_usimplex_float
    wide/wide_opnoise_usimplex_deriv_float
    wide/wide_opnoise_usimplex_Vec3
    wide/wide_opnoise_usimplex_deriv_Vec3
    wide/wide_opstring
    wide/wide_optranscendental
    wide/wide_optrigonometric
    wide/wide_shadingsys
   ../liboslnoise/wide/wide_gabor_anisotropic_disabled
   ../liboslnoise/wide/wide_gabor_anisotropic_enabled
   ../liboslnoise/wide/wide_gabor_hybrid_disabled
   ../liboslnoise/wide/wide_gabor_hybrid_enabled
   ../liboslnoise/wide/wide_gabor_isotropic_disabled
   ../liboslnoise/wide/wide_gabor_isotropic_enabled   
   ../liboslnoise/wide/wide_gabor3_anisotropic_disabled
   ../liboslnoise/wide/wide_gabor3_anisotropic_enabled
   ../liboslnoise/wide/wide_gabor3_hybrid_disabled
   ../liboslnoise/wide/wide_gabor3_hybrid_enabled
   ../liboslnoise/wide/wide_gabor3_isotropic_disabled
   ../liboslnoise/wide/wide_gabor3_isotropic_enabled   
    )


set ( liboslexec_override_limits
    wide/wide_opnoise_periodic_perlin_deriv_float
    wide/wide_opnoise_periodic_perlin_deriv_Vec3
    wide/wide_opnoise_periodic_perlin_float
    wide/wide_opnoise_periodic_perlin_Vec3
    wide/wide_opnoise_periodic_uperlin_deriv_float
    wide/wide_opnoise_periodic_uperlin_deriv_Vec3
    wide/wide_opnoise_periodic_uperlin_float
    wide/wide_opnoise_periodic_uperlin_Vec3
    wide/wide_opnoise_perlin_deriv_float
    wide/wide_opnoise_perlin_deriv_Vec3
    wide/wide_opnoise_perlin_float
    wide/wide_opnoise_perlin_Vec3
    wide/wide_opnoise_uperlin_deriv_float
    wide/wide_opnoise_uperlin_deriv_Vec3
    wide/wide_opnoise_uperlin_float
    wide/wide_opnoise_uperlin_Vec3
   ../liboslnoise/wide/wide_gabor3_anisotropic_disabled
   ../liboslnoise/wide/wide_gabor3_anisotropic_enabled
   ../liboslnoise/wide/wide_gabor3_hybrid_disabled
   ../liboslnoise/wide/wide_gabor3_hybrid_enabled
   ../liboslnoise/wide/wide_gabor3_isotropic_disabled
   ../liboslnoise/wide/wide_gabor3_isotropic_enabled   
    )
    
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
    )
if (BUILD_BATCHED)
    list(APPEND lib_src 
          batched_analysis.cpp
          batched_backendllvm.cpp
          batched_llvm_gen.cpp
          batched_llvm_instance.cpp
          batched_rendservices.cpp
        )
endif()    

if (BUILD_SHARED_LIBS)
    # oslcomp symbols used in oslexec
    list(APPEND lib_src
        ../liboslcomp/ast.cpp
        ../liboslcomp/codegen.cpp
        ../liboslcomp/oslcomp.cpp
        ../liboslcomp/symtab.cpp
        ../liboslcomp/typecheck.cpp
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
    list (TRANSFORM OPENEXR_INCLUDES PREPEND -I
          OUTPUT_VARIABLE ALL_OPENEXR_INCLUDES)
    list (TRANSFORM OpenImageIO_INCLUDES PREPEND -I
          OUTPUT_VARIABLE ALL_OpenImageIO_INCLUDES)

    # Command to turn the .cpp file into LLVM assembly language .s, into
    # LLVM bitcode .bc, then back into a C++ file with the bc embedded!
    add_custom_command ( OUTPUT ${llvm_bc_cpp}
      COMMAND ${LLVM_BC_GENERATOR}
          ${LLVM_COMPILE_FLAGS}
          "-I${CMAKE_CURRENT_SOURCE_DIR}"
          "-I${CMAKE_SOURCE_DIR}/src/include"
          "-I${CMAKE_BINARY_DIR}/include"
          ${ALL_OpenImageIO_INCLUDES}
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
    APPEND PROPERTY COMPILE_DEFINITIONS ${BATCHED_SUPPORT_DEFINES}) 

set (BATCHED_TARGET_LIBS)
foreach(batched_target ${BATCHED_TARGET_LIST})
    set (batched_target_lib "_${batched_target}_oslexec")
    list (APPEND BATCHED_TARGET_LIBS ${batched_target_lib})
    string (REPLACE "_" ";" TARGET_OPTS ${batched_target})
    list (GET TARGET_OPTS 0 TARGET_OPT_SIZE)
    list (GET TARGET_OPTS 1 TARGET_OPT_ISA)
    list (LENGTH TARGET_OPTS NUM_TARGET_OPTS)
    set (TARGET_OPT_FMA "FMA")
    set (TARGET_ISA ${TARGET_OPT_ISA})
    if (NUM_TARGET_OPTS EQUAL 3)
        list (GET TARGET_OPTS 2 TARGET_OPT_FMA)
        set (TARGET_ISA "${TARGET_OPT_ISA}_${TARGET_OPT_FMA}")
    endif() 
    string (SUBSTRING ${TARGET_OPT_SIZE} 1 -1 TARGET_BATCH_SIZE)
    # Strategy is to make a copy of each cpp in liboslexec_target_srcs for 
    # each target batch width and ISA combination, applying compiler flags to define 
    # -D__OSL_WIDTH=[4|8|16]
    # -D__OSL_TARGET_ISA=[AVX512|AVX2|AVX|SSE4_2]
    # and compiler flags to choose correct target ISA for your compiler
    # You may then add/remove the desired  ${TARGET_SOURCES_B[4|8|16]_[AVX512|AVX2|AVX|SSE4_2]}
    # to your add_library call 
    set (TARGET_LIB_SOURCES)
    set (TARGET_CXX_DEFS)
    set (TARGET_CXX_OPTS)
    
    list (APPEND TARGET_CXX_OPTS
        "-I${CMAKE_CURRENT_SOURCE_DIR}/wide"
        "-I${CMAKE_CURRENT_SOURCE_DIR}/../liboslnoise/wide"
        )
        
    list (APPEND TARGET_CXX_DEFS
        "__OSL_WIDTH=${TARGET_BATCH_SIZE}"
        "__OSL_TARGET_ISA=${TARGET_ISA}"        
        "OSL_OPENMP_SIMD"        
        )

    if (CMAKE_COMPILER_IS_INTEL)
        if (MSVC)
            if (${TARGET_OPT_ISA} STREQUAL "AVX512")
                list (APPEND TARGET_CXX_OPTS "/QxCORE-AVX512")
                if (${TARGET_BATCH_SIZE} STREQUAL "16")
                    list (APPEND TARGET_CXX_OPTS "/Qopt-zmm-usage:high")
                else ()
                    list (APPEND TARGET_CXX_OPTS "/Qopt-zmm-usage:low")
                endif ()
            elseif (${TARGET_OPT_ISA} STREQUAL "AVX2")
                list (APPEND TARGET_CXX_OPTS "/QxCORE-AVX2")
            elseif (${TARGET_OPT_ISA} STREQUAL "AVX")
                list (APPEND TARGET_CXX_OPTS "/QxAVX")
            else ()
                message (FATAL_ERROR "Unknown ISA=${TARGET_OPT_ISA} extract from USE_BATCHED entry ${batched_target}")
            endif ()
            
            if (${TARGET_OPT_FMA} STREQUAL "noFMA")
                list (APPEND TARGET_CXX_OPTS "/Qfma-")
            endif ()
            
            list (APPEND TARGET_CXX_OPTS 
                "/Qprec-div" 
                "/Qopt-report:5"
                "/Qopenmp-simd"
                )
                
        else ()
            if (${TARGET_OPT_ISA} STREQUAL "AVX512")
                list (APPEND TARGET_CXX_OPTS "-xCORE-AVX512")
                if (${TARGET_BATCH_SIZE} STREQUAL "16")
                    list (APPEND TARGET_CXX_OPTS "-qopt-zmm-usage:high")
                else ()
                    list (APPEND TARGET_CXX_OPTS "-qopt-zmm-usage:low")
                endif ()
            elseif (${TARGET_OPT_ISA} STREQUAL "AVX2")
                list (APPEND TARGET_CXX_OPTS "-xCORE-AVX2")
            elseif (${TARGET_OPT_ISA} STREQUAL "AVX")
                list (APPEND TARGET_CXX_OPTS "-xAVX")
            else ()
                message (FATAL_ERROR "Unknown ISA=${TARGET_OPT_ISA} extract from USE_BATCHED entry ${batched_target}")
            endif ()
            
            if (${TARGET_OPT_FMA} STREQUAL "noFMA")
                list (APPEND TARGET_CXX_OPTS "-no-fma")
            endif ()
            
            list (APPEND TARGET_CXX_OPTS 
                "-prec-div" 
                "-qopt-report=5" 
                "-qopenmp-simd"
                )
                
        endif ()
    endif ()
    
    if (CMAKE_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_APPLECLANG)
        # REQUIRES CLANG 7+ to use sucessfully vectorize with OpenMP simd
        
        if (${TARGET_OPT_ISA} STREQUAL "AVX512")
            list (APPEND TARGET_CXX_OPTS "-march=skylake-avx512")
            if (${TARGET_BATCH_SIZE} STREQUAL "16")
                list (APPEND TARGET_CXX_OPTS "-mprefer-vector-width=512")
            else ()
                list (APPEND TARGET_CXX_OPTS "-mprefer-vector-width=256")
            endif ()
        elseif (${TARGET_OPT_ISA} STREQUAL "AVX2")
            list (APPEND TARGET_CXX_OPTS "-march=core-avx2")
        elseif (${TARGET_OPT_ISA} STREQUAL "AVX")
            list (APPEND TARGET_CXX_OPTS "-march=corei7-avx")
        else ()
            message (FATAL_ERROR "Unknown ISA=${TARGET_OPT_ISA} extract from USE_BATCHED entry ${batched_target}")
        endif ()
        
        if (${TARGET_OPT_FMA} STREQUAL "noFMA")
            list (APPEND TARGET_CXX_OPTS "-mno-fma")
        endif ()
        
        # large SIMD function loops will exceed llvm's -inline-threshold default of 225.
        # remark: loop not vectorized: call instruction cannot be vectorized [-Rpass-analysis]     
        # choose to increase that limit via compiler flags vs. 
        # workaround with __attribute__((flatten))     
        list (APPEND TARGET_CXX_OPTS "-mllvm" "-inline-threshold=100000")
            
        # For loops with small loop bodies, clang was unrolling the loop before
        # #pragma omp simd
        # was processed.  Thus there was no loop left for the pragma to apply to
        # To work around this, for these wide SIMD libraries, we will disable
        # clang's loop unrolling.  We really don't want to do this and wish to remove
        # it as soon as clang addresses this issue.
        list (APPEND TARGET_CXX_OPTS "-fno-unroll-loops")

        list (APPEND TARGET_CXX_OPTS "-fopenmp")
    endif ()
    
    
    if (CMAKE_COMPILER_IS_GNUCC)
        # Experimental, not exhaustively tested with GCC.
        # More compiler options may be needed to successfully vectorize
        # complex control flow 
        list (APPEND TARGET_CXX_OPTS "-finline-limit=1000000")
        
        if (${TARGET_OPT_ISA} STREQUAL "AVX512")
            list (APPEND TARGET_CXX_OPTS "-march=skylake-avx512")
            if (NOT ${GCC_VERSION} VERSION_LESS 8.1)
                if (${TARGET_BATCH_SIZE} STREQUAL "16")
                    list (APPEND TARGET_CXX_OPTS "-mprefer-vector-width=512")
                else ()
                    list (APPEND TARGET_CXX_OPTS "-mprefer-vector-width=256")
                endif ()
            endif ()
        elseif (${TARGET_OPT_ISA} STREQUAL "AVX2")
            list (APPEND TARGET_CXX_OPTS "-march=haswell")
        elseif (${TARGET_OPT_ISA} STREQUAL "AVX")
            list (APPEND TARGET_CXX_OPTS "-march=sandybridge")
        else ()
            message (FATAL_ERROR "Unknown ISA=${TARGET_OPT_ISA} extract from USE_BATCHED entry ${batched_target}")
        endif ()
        
        if (${TARGET_OPT_FMA} STREQUAL "noFMA")
            list (APPEND TARGET_CXX_OPTS "-mno-fma")
        endif ()
        
        list (APPEND TARGET_CXX_OPTS "-fopenmp-simd")
    endif ()    
        
    foreach(target_src ${liboslexec_target_srcs})
        set(TARGET_SRC "${CMAKE_CURRENT_BINARY_DIR}/${target_src}_${batched_target}.cpp")
        set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/${target_src}.cpp")        
        CONFIGURE_FILE("${SRC}" ${TARGET_SRC} COPYONLY)
        list(APPEND TARGET_LIB_SOURCES "${TARGET_SRC}")
        
        set_property(SOURCE ${TARGET_SRC}
            APPEND PROPERTY COMPILE_DEFINITIONS ${TARGET_CXX_DEFS})
            
        set_property(SOURCE ${TARGET_SRC}  
            APPEND PROPERTY COMPILE_OPTIONS ${TARGET_CXX_OPTS})
            
        if (${target_src} IN_LIST liboslexec_override_limits)
            if (CMAKE_COMPILER_IS_INTEL)
                if (MSVC)
                    set_property(SOURCE ${TARGET_SRC} APPEND PROPERTY COMPILE_OPTIONS "/Qoverride-limits")
                else ()
                    set_property(SOURCE ${TARGET_SRC} APPEND PROPERTY COMPILE_OPTIONS "-qoverride-limits")
                endif()
            endif()
        endif()
    endforeach(target_src)
    
    add_library ( ${batched_target_lib} MODULE ${TARGET_LIB_SOURCES} )
      
    target_include_directories (${batched_target_lib}
        PUBLIC
            ${CMAKE_INSTALL_FULL_INCLUDEDIR}
            ${IMATH_INCLUDES}
        )
    target_link_libraries (${batched_target_lib}
        PUBLIC
            OpenImageIO::OpenImageIO
        )
endforeach(batched_target)

add_library (${local_lib} ${lib_src})
target_include_directories (${local_lib}
    PUBLIC
        ${CMAKE_INSTALL_FULL_INCLUDEDIR}
        ${IMATH_INCLUDES}
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
        oslquery oslnoise
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
        ${Boost_LIBRARIES} ${CMAKE_DL_LIBS}
        ${CLANG_LIBRARIES}
        ${LLVM_LIBRARIES} ${LLVM_LDFLAGS} ${LLVM_SYSTEM_LIBRARIES}
    )

target_compile_features (${local_lib}
                         PUBLIC cxx_std_${CMAKE_CXX_STANDARD})

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

foreach(batched_target_lib ${BATCHED_TARGET_LIBS})
    install_targets ( ${batched_target_lib} )
endforeach(batched_target_lib)

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
