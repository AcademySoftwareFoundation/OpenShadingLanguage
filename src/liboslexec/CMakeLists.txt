# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# NOTE: for any additions/deletions from liboslexec_target_srcs,
# please update wide_target_combine_text_and_rodata.ld
set ( liboslexec_target_srcs
    wide/wide_opalgebraic
    wide/wide_opcolor
    wide/wide_opclosure
    wide/wide_opdictionary
    wide/wide_ophash
    wide/wide_opmatrix
    wide/wide_opmessage
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
    wide/wide_oppointcloud
    wide/wide_opspline
    wide/wide_opstring
    wide/wide_optest_float
    wide/wide_optexture
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
    wide/wide_opspline
   ../liboslnoise/wide/wide_gabor3_anisotropic_disabled
   ../liboslnoise/wide/wide_gabor3_anisotropic_enabled
   ../liboslnoise/wide/wide_gabor3_hybrid_disabled
   ../liboslnoise/wide/wide_gabor3_hybrid_enabled
   ../liboslnoise/wide/wide_gabor3_isotropic_disabled
   ../liboslnoise/wide/wide_gabor3_isotropic_enabled   
    )
    
set ( liboslexec_require_INF_NaN
    shadingsys.cpp
    wide/wide_shadingsys
    wide/wide_optest_float
    )

set ( liboslexec_exceeds_INTELCLANG_inlining_limits
    wide/wide_opspline
    )

set (local_lib oslexec)
set (lib_src
          shadingsys.cpp closure.cpp
          dictionary.cpp
          context.cpp instance.cpp
          loadshader.cpp master.cpp
          opcolor.cpp opfmt.cpp opmatrix.cpp opmessage.cpp
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
          rs_fallback.cpp
          journal.cpp
    )
if (OSL_BUILD_BATCHED)
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
if (BUILD_SHARED_LIBS)
    FLEX_BISON ( ../liboslcomp/osllex.l ../liboslcomp/oslgram.y osl lib_src compiler_headers )
endif()

set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS" )


macro ( REQUIRE_INF_NAN SRC )
    if (CMAKE_COMPILER_IS_INTELCLANG)
        # if -ffast-math is enabled to to the underlying compiler, 
        # inf's and NaN's may not be handled properly (by design)
        if (MSVC)
            set_property(SOURCE ${SRC} APPEND PROPERTY COMPILE_OPTIONS "/fp:precise")
            #message (STATUS "SOURCE ${SRC} APPEND PROPERTY COMPILE_OPTIONS /fp:precise")
        else ()
            #set_property(SOURCE $SRC} APPEND PROPERTY COMPILE_OPTIONS "-fp-model=precise")
            # Disabling fast-math should be enough to enable proper support of INF and NaN
            #set_property(SOURCE ${SRC} APPEND PROPERTY COMPILE_OPTIONS "-fno-fast-math")
            # We can explicitly honor inf, nans, and signed 0's
            set_property(SOURCE ${SRC} APPEND PROPERTY COMPILE_OPTIONS 
                "-fhonor-infinities" 
                "-fhonor-nans" 
                "-fsigned-zeros")
        endif()
    endif()
endmacro ( )

# Apply complilation options as needed
foreach(src ${lib_src})
    if (${src} IN_LIST liboslexec_require_INF_NaN)
        set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        REQUIRE_INF_NAN ( ${SRC} )  
    endif()
endforeach(src)

if (USE_LLVM_BITCODE)
    set (llvm_ops_srcs llvm_ops.cpp)

    set(include_dirs ${CMAKE_CURRENT_SOURCE_DIR})
    list(APPEND include_dirs ${CMAKE_SOURCE_DIR}/src/include)
    list(APPEND include_dirs ${CMAKE_BINARY_DIR}/include)
    list(APPEND include_dirs ${IMATH_INCLUDES})
    list(APPEND include_dirs ${OPENEXR_INCLUDES})
    list(APPEND include_dirs ${OpenImageIO_INCLUDES})
    list(APPEND include_dirs ${Boost_INCLUDE_DIRS})

    EMBED_LLVM_BITCODE_IN_CPP ( "${llvm_ops_srcs}" "_host" "osl_llvm_compiled_ops" lib_src "" "${include_dirs}")

    set (rs_dependent_ops_srcs
         opmatrix.cpp opfmt.cpp
         )
    # Achieve the effect of absorbing osl_llvm_compiled_ops by adding its 
    # sources to rs_dependent_ops_srcs which avoids having to do it at runtime.
    list (APPEND rs_dependent_ops_srcs ${llvm_ops_srcs})

    EMBED_LLVM_BITCODE_IN_CPP ( "${rs_dependent_ops_srcs}" "_host_rs" "osl_llvm_compiled_rs_dependent_ops" lib_src "" "${include_dirs}")

    if (CUDA_FOUND)
        add_definitions (-DOSL_LLVM_CUDA_BITCODE)

        # The shadeops are compiled to both LLVM bitcode and PTX, and embedded
        # in liboslexec in binary format. The bitcode is used to seed the LLVM
        # module for each shader, and the PTX can be retrieved by the renderer
        # through the shading system.
        set (liboslexec_shadeops_srcs
            ${CMAKE_SOURCE_DIR}/src/liboslexec/llvm_ops.cpp
            ${CMAKE_SOURCE_DIR}/src/liboslexec/opnoise.cpp
            ${CMAKE_SOURCE_DIR}/src/liboslexec/opspline.cpp
            ${CMAKE_SOURCE_DIR}/src/liboslexec/opcolor.cpp
            ${CMAKE_SOURCE_DIR}/src/liboslexec/opmatrix.cpp
            ${CMAKE_SOURCE_DIR}/src/liboslnoise/gabornoise.cpp
            ${CMAKE_SOURCE_DIR}/src/liboslnoise/simplexnoise.cpp
        )

        CUDA_SHADEOPS_COMPILE ( "shadeops_cuda"
            shadeops_cuda_bc
            shadeops_cuda_ptx
            "${liboslexec_shadeops_srcs}"
            ""
        )

        # Serialize the linked bitcode into a CPP file to be embedded in the current target binary
        set (shadeops_bc_cuda_cpp "${CMAKE_CURRENT_BINARY_DIR}/shadeops_cuda.bc.cpp")
        MAKE_EMBEDDED_CPP( "shadeops_cuda_llvm_compiled_ops" ${shadeops_bc_cuda_cpp} ${shadeops_cuda_bc} )
        list (APPEND lib_src ${shadeops_bc_cuda_cpp})

        # Serialize the linked PTX into a CPP file to be embedded in the current target binary
        set (shadeops_ptx_cuda_cpp "${CMAKE_CURRENT_BINARY_DIR}/shadeops_cuda.ptx.cpp")
        MAKE_EMBEDDED_CPP( "shadeops_cuda_ptx_compiled_ops" ${shadeops_ptx_cuda_cpp} ${shadeops_cuda_ptx} )
        list (APPEND lib_src ${shadeops_ptx_cuda_cpp})

        # Installing the shadeops PTX isn't necessary since the renderer can
        # retrieve it through the shading system. But for testing purposes it is
        # handy to make the PTX available offline.
        install (FILES ${shadeops_cuda_ptx}
            DESTINATION ${OSL_PTX_INSTALL_DIR})

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
                "/Qopenmp-simd"
                )
            if (VEC_REPORT)
                list (APPEND TARGET_CXX_OPTS "/Qopt-report:5")
            endif ()
                
                
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
                "-qopenmp-simd"
                "-qno-opt-multiple-gather-scatter-by-shuffles"
                )
                
            if (VEC_REPORT)
                list (APPEND TARGET_CXX_OPTS "-qopt-report=5")
            endif ()
                
        endif ()
    endif ()
    
    if (CMAKE_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_APPLECLANG)
        # REQUIRES CLANG 7+ to use successfully vectorize with OpenMP simd
        
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
            # Do NOT use "-mno-fma" as it may lower ISA to one without any FMA
            # instructions (AVX512->AVX2).  Our intent is to not generate 
            # FMA's but still use the available instructions and register 
            # width. However math library calls may use still end up using 
            # FMA's, but lets try turning of fused math ops because many OSL 
            # math functions are from OIIO/IMath headers generate non FMA
            list (APPEND TARGET_CXX_OPTS "-ffp-contract=off")
        endif ()
        
        if (NOT CMAKE_COMPILER_IS_INTELCLANG)
            # large SIMD function loops will exceed llvm's -inline-threshold default of 225.
            # remark: loop not vectorized: call instruction cannot be vectorized [-Rpass-analysis]     
            # choose to increase that limit via compiler flags vs. 
            # workaround with __attribute__((flatten))     
            list (APPEND TARGET_CXX_OPTS "-mllvm" "-inline-threshold=100000")
            # NOTE: INTELCLANG we use icx/icc specific "#pragma force inline recursive"
            #       to only inline the SIMD loops contents vs. everything.
        endif ()
            
        # For loops with small loop bodies, clang was unrolling the loop before
        # #pragma omp simd
        # was processed.  Thus there was no loop left for the pragma to apply to
        # To work around this, for these wide SIMD libraries, we will disable
        # clang's loop unrolling.  We really don't want to do this and wish to remove
        # it as soon as clang addresses this issue.
        list (APPEND TARGET_CXX_OPTS "-fno-unroll-loops")

        if (CMAKE_COMPILER_IS_INTELCLANG)
            # utilize Intel compiler's optimized openmp simd generation vs. open source "-qopenmp-simd"
            list (APPEND TARGET_CXX_OPTS "-fiopenmp-simd")
        else ()
            list (APPEND TARGET_CXX_OPTS "-fopenmp")
        endif ()
        
        if (VEC_REPORT)
            if (CMAKE_COMPILER_IS_INTELCLANG)
                list (APPEND TARGET_CXX_OPTS 
                    "-qopt-report=3"
                    )
            else ()
                list (APPEND TARGET_CXX_OPTS 
                    "-Rpass=loop-vectorize" 
                    "-Rpass-analysis=loop-vectorize" 
                    # Uncomment next section to investigate failures in vectorization
                    #"-Rpass-missed=loop-vectorize"
                    )
            endif ()
        endif ()
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
            # Do NOT use "-mno-fma" as it may lower ISA to one without any FMA
            # instructions (AVX512->AVX2).  Our intent is to not generate 
            # FMA's but still use the available instructions and register 
            # width. However math library calls may use still end up using 
            # FMA's, but lets try turning of fused math ops because many OSL 
            # math functions are from OIIO/IMath headers generate non FMA
            list (APPEND TARGET_CXX_OPTS "-ffp-contract=off")
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
        
        if (${target_src} IN_LIST liboslexec_require_INF_NaN)
            REQUIRE_INF_NAN ( ${TARGET_SRC} ) 
        endif()

        if (CMAKE_COMPILER_IS_INTELCLANG)
            if (${target_src} IN_LIST liboslexec_exceeds_INTELCLANG_inlining_limits)
                # NOTE: Some files still require increasing
                #       inlining threshold to successfully vectorize.
                #       Retest with future compiler versions
                set_property(SOURCE ${TARGET_SRC} APPEND PROPERTY COMPILE_OPTIONS 
                    "-mllvm" "-inline-threshold=100000")
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
    if (partio_FOUND)
        target_link_libraries(${batched_target_lib} PRIVATE partio::partio)
        target_compile_definitions (${batched_target_lib} PRIVATE USE_PARTIO=1)
    endif ()
        
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
        $<TARGET_NAME_IF_EXISTS:Imath::Imath>
        $<TARGET_NAME_IF_EXISTS:Imath::Half>
        # For OpenEXR >= 2.4/2.5 with reliable exported targets
        $<TARGET_NAME_IF_EXISTS:IlmBase::Imath>
        $<TARGET_NAME_IF_EXISTS:IlmBase::Half>
        $<TARGET_NAME_IF_EXISTS:IlmBase::IlmThread>
        $<TARGET_NAME_IF_EXISTS:IlmBase::Iex>
        # For OpenEXR <= 2.3:
        ${ILMBASE_LIBRARIES}
    PRIVATE
        pugixml::pugixml
        ${Boost_LIBRARIES} ${CMAKE_DL_LIBS}
        ${CLANG_LIBRARIES}
        ${LLVM_LIBRARIES} ${LLVM_LDFLAGS} ${LLVM_SYSTEM_LIBRARIES}
    )

target_compile_features (${local_lib}
                         INTERFACE cxx_std_${DOWNSTREAM_CXX_STANDARD})

set (link_flags "${VISIBILITY_MAP_COMMAND} ${EXTRA_DSO_LINK_ARGS}")
if (UNIX AND NOT APPLE)
    # Hide symbols from any static dependent libraries embedded here.
    set (link_flags "${link_flags} -Wl,--exclude-libs,ALL")
endif ()

set_target_properties (${local_lib}
        PROPERTIES
        VERSION     ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
        SOVERSION   ${SOVERSION}
        OUTPUT_NAME ${local_lib}${OSL_LIBNAME_SUFFIX}
        LINK_FLAGS  ${link_flags}
    )

# add_dependencies (${local_lib} "${CMAKE_SOURCE_DIR}/src/build-scripts/hidesymbols.map")

foreach(batched_target_lib ${BATCHED_TARGET_LIBS})
    install_targets ( ${batched_target_lib} )
endforeach(batched_target_lib)

install_targets (${local_lib})

# Unit tests
if (OSL_BUILD_TESTS AND BUILD_TESTING)
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
