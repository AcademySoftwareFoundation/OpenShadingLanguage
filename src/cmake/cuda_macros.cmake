# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

set ($OSL_EXTRA_NVCC_ARGS "" CACHE STRING "Custom args passed to nvcc when compiling CUDA code")

# Compile a CUDA file to PTX using NVCC
function ( NVCC_COMPILE cuda_src ptx_generated extra_nvcc_args )
    get_filename_component ( cuda_src_we ${cuda_src} NAME_WE )
    get_filename_component ( cuda_src_dir ${cuda_src} DIRECTORY )
    set (cuda_ptx "${CMAKE_CURRENT_BINARY_DIR}/${cuda_src_we}.ptx" )
    set (${ptxlist} ${${ptxlist}} ${cuda_ptx} )
    set (${ptx_generated} ${cuda_ptx} PARENT_SCOPE)
    file ( GLOB cuda_headers "${cuda_src_dir}/*.h" )

    add_custom_command ( OUTPUT ${cuda_ptx}
        COMMAND ${CUDA_NVCC_EXECUTABLE}
            "-I${OPTIX_INCLUDES}"
            "-I${CUDA_INCLUDES}"
            "-I${CMAKE_CURRENT_SOURCE_DIR}"
            "-I${CMAKE_BINARY_DIR}/include"
            "-I${PROJECT_SOURCE_DIR}/src/include"
            "-I${PROJECT_SOURCE_DIR}/src/cuda_common"
            "-I${OpenImageIO_INCLUDES}"
            "-I${ILMBASE_INCLUDES}"
            "-I${Boost_INCLUDE_DIRS}"
            "-DFMT_DEPRECATED=\"\""
            ${LLVM_COMPILE_FLAGS}
            -DOSL_USE_FAST_MATH=1
            -m64 -arch ${CUDA_TARGET_ARCH} -ptx
            --std=c++14 -dc -O3 --use_fast_math --expt-relaxed-constexpr
            ${extra_nvcc_args}
            ${OSL_EXTRA_NVCC_ARGS}
            ${cuda_src} -o ${cuda_ptx}
        MAIN_DEPENDENCY ${cuda_src}
        DEPENDS ${cuda_src} ${cuda_headers} oslexec
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
endfunction ()

# Function to compile a C++ source file to CUDA-compatible LLVM bitcode
function ( MAKE_CUDA_BITCODE src suffix generated_bc extra_clang_args )
    get_filename_component ( src_we ${src} NAME_WE )
    set (asm_cuda "${CMAKE_CURRENT_BINARY_DIR}/${src_we}${suffix}.s" )
    set (bc_cuda "${CMAKE_CURRENT_BINARY_DIR}/${src_we}${suffix}.bc" )
    set (${generated_bc} ${bc_cuda} PARENT_SCOPE )

    # Setup the compile flags
    get_property (CURRENT_DEFINITIONS DIRECTORY PROPERTY COMPILE_DEFINITIONS)
    if (VERBOSE)
        message (STATUS "Current #defines are ${CURRENT_DEFINITIONS}")
    endif ()
    foreach (def ${CURRENT_DEFINITIONS})
        set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} "-D${def}")
    endforeach()
    set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} ${SIMD_COMPILE_FLAGS} ${CSTD_FLAGS})

    # Setup the bitcode generator
    if (NOT LLVM_BC_GENERATOR)
        FIND_PROGRAM(LLVM_BC_GENERATOR NAMES "clang++" PATHS "${LLVM_DIRECTORY}/bin" NO_DEFAULT_PATH NO_CMAKE_SYSTEM_PATH NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_PATH)
    endif ()
    # If that didn't work, look anywhere
    if (NOT LLVM_BC_GENERATOR)
        # Wasn't in their build, look anywhere
        FIND_PROGRAM(LLVM_BC_GENERATOR NAMES clang++ llvm-g++)
    endif ()

    if (NOT LLVM_BC_GENERATOR)
        message (FATAL_ERROR "You must have a valid llvm bitcode generator (clang++) somewhere.")
    endif ()
    if (VERBOSE)
        message (STATUS "Using LLVM_BC_GENERATOR ${LLVM_BC_GENERATOR} to generate bitcode.")
    endif()

    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        # fix compilation error when using MSVC
        set (CLANG_MSVC_FIX "-DBOOST_CONFIG_REQUIRES_THREADS_HPP")

        # these are warnings triggered by the dllimport/export attributes not being supported when
        # compiling for Cuda. When all 3rd parties have their export macro fixed these warnings
        # can be restored.
        set (CLANG_MSVC_FIX "${CLANG_MSVC_FIX} -Wno-ignored-attributes -Wno-unknown-attributes")
    endif ()

    add_custom_command (OUTPUT ${bc_cuda}
        COMMAND ${LLVM_BC_GENERATOR}
            "-I${OPTIX_INCLUDES}"
            "-I${CUDA_INCLUDES}"
            "-I${CMAKE_CURRENT_SOURCE_DIR}"
            "-I${CMAKE_BINARY_DIR}/include"
            "-I${PROJECT_SOURCE_DIR}/src/include"
            "-I${PROJECT_SOURCE_DIR}/src/cuda_common"
            "-I${OpenImageIO_INCLUDES}"
            "-I${ILMBASE_INCLUDE_DIR}"
            "-I${Boost_INCLUDE_DIRS}"
            ${LLVM_COMPILE_FLAGS} ${CUDA_LIB_FLAGS} ${CLANG_MSVC_FIX}
            -D__CUDACC__ -DOSL_COMPILING_TO_BITCODE=1 -DNDEBUG -DOIIO_NO_SSE -D__CUDADEVRT_INTERNAL__
            --language=cuda --cuda-device-only --cuda-gpu-arch=${CUDA_TARGET_ARCH}
            -Wno-deprecated-register -Wno-format-security
            -O3 -fno-math-errno -ffast-math -S -emit-llvm ${extra_clang_args}
            ${src} -o ${asm_cuda}
        COMMAND "${LLVM_DIRECTORY}/bin/llvm-as" -f -o ${bc_cuda} ${asm_cuda}
        DEPENDS ${exec_headers} ${PROJECT_PUBLIC_HEADERS} ${src}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
endfunction ()

# Compile a CUDA source file to LLVM bitcode, and then serialize the bitcode to
# a C++ file to be compiled into the target executable.
function ( LLVM_COMPILE_CUDA llvm_src headers prefix llvm_bc_cpp_generated extra_clang_args )
    get_filename_component (llvmsrc_we ${llvm_src} NAME_WE)
    set (llvm_bc_cpp "${CMAKE_CURRENT_BINARY_DIR}/${llvmsrc_we}.bc.cpp")
    set (${llvm_bc_cpp_generated} ${llvm_bc_cpp} PARENT_SCOPE)

    MAKE_CUDA_BITCODE (${llvm_src} "" llvm_bc "${extra_clang_args}")

    add_custom_command (OUTPUT ${llvm_bc_cpp}
        COMMAND python "${CMAKE_SOURCE_DIR}/src/liboslexec/serialize-bc.py" ${llvm_bc} ${llvm_bc_cpp} ${prefix}
        MAIN_DEPENDENCY ${llvm_src}
        DEPENDS "${CMAKE_SOURCE_DIR}/src/liboslexec/serialize-bc.py" ${llvm_src} ${headers} ${llvm_bc}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
endfunction ()
