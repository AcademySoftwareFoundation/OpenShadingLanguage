# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

set ($OSL_EXTRA_NVCC_ARGS "" CACHE STRING "Custom args passed to nvcc when compiling CUDA code")
set (CUDA_OPT_FLAG_NVCC "-O3" CACHE STRING "The optimization level to use when compiling CUDA/C++ files with nvcc")
set (CUDA_OPT_FLAG_CLANG "-O3" CACHE STRING "The optimization level to use when compiling CUDA/C++ files with clang")

# Compile a CUDA file to PTX using NVCC
function ( NVCC_COMPILE cuda_src extra_headers ptx_generated extra_nvcc_args )
    get_filename_component ( cuda_src_we ${cuda_src} NAME_WE )
    get_filename_component ( cuda_src_dir ${cuda_src} DIRECTORY )
    set (cuda_ptx "${CMAKE_CURRENT_BINARY_DIR}/${cuda_src_we}.ptx" )
    set (${ptxlist} ${${ptxlist}} ${cuda_ptx} )
    set (${ptx_generated} ${cuda_ptx} PARENT_SCOPE)
    file ( GLOB cuda_headers "${cuda_src_dir}/*.h" )
    list (APPEND cuda_headers ${extra_headers})

    list (TRANSFORM IMATH_INCLUDES PREPEND -I
          OUTPUT_VARIABLE ALL_IMATH_INCLUDES)
    list (TRANSFORM OPENEXR_INCLUDES PREPEND -I
          OUTPUT_VARIABLE ALL_OPENEXR_INCLUDES)
    list (TRANSFORM OpenImageIO_INCLUDES PREPEND -I
          OUTPUT_VARIABLE ALL_OpenImageIO_INCLUDES)

    add_custom_command ( OUTPUT ${cuda_ptx}
        COMMAND ${CUDA_NVCC_EXECUTABLE}
            "-I${OPTIX_INCLUDES}"
            "-I${CUDA_INCLUDES}"
            "-I${CMAKE_CURRENT_SOURCE_DIR}"
            "-I${CMAKE_BINARY_DIR}/include"
            "-I${PROJECT_SOURCE_DIR}/src/include"
            "-I${PROJECT_SOURCE_DIR}/src/cuda_common"
            ${ALL_OpenImageIO_INCLUDES}
            ${ALL_IMATH_INCLUDES}
            ${ALL_OPENEXR_INCLUDES}
            "-I${Boost_INCLUDE_DIRS}"
            "-DFMT_DEPRECATED=\"\""
            ${LLVM_COMPILE_FLAGS}
            -DOSL_USE_FAST_MATH=1
            -m64 -arch ${CUDA_TARGET_ARCH} -ptx
            --std=c++14 -dc --use_fast_math ${CUDA_OPT_FLAG_NVCC} --expt-relaxed-constexpr
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

    if (NOT LLVM_LLC_TOOL)
        find_program (LLVM_LLC_TOOL NAMES "llc"
                PATHS "${LLVM_DIRECTORY}/bin" "${LLVM_DIRECTORY}/tools/llvm"
                NO_CMAKE_PATH NO_DEFAULT_PATH NO_CMAKE_SYSTEM_PATH
                NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_ENVIRONMENT_PATH)
    endif ()

    if (NOT LLVM_OPT_TOOL)
        find_program (LLVM_OPT_TOOL NAMES "opt"
                PATHS "${LLVM_DIRECTORY}/bin" "${LLVM_DIRECTORY}/tools/llvm"
                NO_CMAKE_PATH NO_DEFAULT_PATH NO_CMAKE_SYSTEM_PATH
                NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_ENVIRONMENT_PATH)
    endif ()

    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        # fix compilation error when using MSVC
        set (CLANG_MSVC_FIX "-DBOOST_CONFIG_REQUIRES_THREADS_HPP")

        # these are warnings triggered by the dllimport/export attributes not being supported when
        # compiling for Cuda. When all 3rd parties have their export macro fixed these warnings
        # can be restored.
        set (CLANG_MSVC_FIX "${CLANG_MSVC_FIX} -Wno-ignored-attributes -Wno-unknown-attributes")
    endif ()

    if (${LLVM_VERSION} VERSION_GREATER_EQUAL 15.0)
        # Until we fully support opaque pointers, we need to disable
        # them when using LLVM 15.
        list (APPEND LLVM_COMPILE_FLAGS -Xclang -no-opaque-pointers)
    endif ()

    list (TRANSFORM IMATH_INCLUDES PREPEND -I
          OUTPUT_VARIABLE ALL_IMATH_INCLUDES)
    list (TRANSFORM OPENEXR_INCLUDES PREPEND -I
          OUTPUT_VARIABLE ALL_OPENEXR_INCLUDES)
    list (TRANSFORM OpenImageIO_INCLUDES PREPEND -I
          OUTPUT_VARIABLE ALL_OpenImageIO_INCLUDES)

    add_custom_command (OUTPUT ${bc_cuda}
        COMMAND ${LLVM_BC_GENERATOR}
            "-I${OPTIX_INCLUDES}"
            "-I${CUDA_INCLUDES}"
            "-I${CMAKE_CURRENT_SOURCE_DIR}"
            "-I${CMAKE_SOURCE_DIR}/src/liboslexec"
            "-I${CMAKE_BINARY_DIR}/include"
            "-I${PROJECT_SOURCE_DIR}/src/include"
            "-I${PROJECT_SOURCE_DIR}/src/cuda_common"
            ${ALL_OpenImageIO_INCLUDES}
            ${ALL_IMATH_INCLUDES}
            ${ALL_OPENEXR_INCLUDES}
            "-I${Boost_INCLUDE_DIRS}"
            ${LLVM_COMPILE_FLAGS} ${CUDA_LIB_FLAGS} ${CLANG_MSVC_FIX}
            -D__CUDACC__ -DOSL_COMPILING_TO_BITCODE=1 -DNDEBUG -DOIIO_NO_SSE -D__CUDADEVRT_INTERNAL__
            --language=cuda --cuda-device-only --cuda-gpu-arch=${CUDA_TARGET_ARCH}
            -Wno-deprecated-register -Wno-format-security
            -fno-math-errno -ffast-math ${CUDA_OPT_FLAG_CLANG} -S -emit-llvm ${extra_clang_args}
            ${src} -o ${asm_cuda}
        COMMAND ${LLVM_AS_TOOL} -f -o ${bc_cuda} ${asm_cuda}
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
        COMMAND ${Python_EXECUTABLE} "${CMAKE_SOURCE_DIR}/src/build-scripts/serialize-bc.py" ${llvm_bc} ${llvm_bc_cpp} ${prefix}
        MAIN_DEPENDENCY ${llvm_src}
        DEPENDS "${CMAKE_SOURCE_DIR}/src/build-scripts/serialize-bc.py" ${llvm_src} ${headers} ${llvm_bc}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
endfunction ()

macro ( CUDA_SHADEOPS_COMPILE srclist rend_lib_src headers )
    # Add all of the "shadeops" sources that need to be compiled to LLVM bitcode for CUDA
    set ( shadeops_srcs
        ${CMAKE_SOURCE_DIR}/src/liboslexec/llvm_ops.cpp
        ${CMAKE_SOURCE_DIR}/src/liboslexec/opnoise.cpp
        ${CMAKE_SOURCE_DIR}/src/liboslexec/opspline.cpp
        ${CMAKE_SOURCE_DIR}/src/liboslexec/opcolor.cpp
        ${CMAKE_SOURCE_DIR}/src/liboslexec/opmatrix.cpp
        ${CMAKE_SOURCE_DIR}/src/liboslnoise/gabornoise.cpp
        ${CMAKE_SOURCE_DIR}/src/liboslnoise/simplexnoise.cpp
        ${rend_lib_src}
        )

    set (shadeops_bc_cuda_cpp   "${CMAKE_CURRENT_BINARY_DIR}/shadeops_cuda.bc.cpp")
    set (linked_shadeops_bc     "${CMAKE_CURRENT_BINARY_DIR}/linked_shadeops.bc")
    set (linked_shadeops_opt_bc "${CMAKE_CURRENT_BINARY_DIR}/linked_shadeops_opt.bc")
    set (linked_shadeops_ptx    "${CMAKE_CURRENT_BINARY_DIR}/linked_shadeops.ptx")

    list (APPEND ${srclist} ${shadeops_bc_cuda_cpp})

    foreach ( shadeops_src ${shadeops_srcs} )
        MAKE_CUDA_BITCODE ( ${shadeops_src} "_cuda" shadeops_bc "" )
        list ( APPEND shadeops_bc_list ${shadeops_bc} )
    endforeach ()

    # Link all of the individual LLVM bitcode files, and emit PTX for the linked bitcode
    add_custom_command ( OUTPUT ${linked_shadeops_opt_bc} ${linked_shadeops_ptx}
        COMMAND ${LLVM_LINK_TOOL} ${shadeops_bc_list} -o ${linked_shadeops_bc}
        COMMAND ${LLVM_OPT_TOOL} ${CUDA_OPT_FLAG_CLANG} ${linked_shadeops_bc} -o ${linked_shadeops_opt_bc}
        COMMAND ${LLVM_LLC_TOOL} --march=nvptx64 -mcpu=${CUDA_TARGET_ARCH} ${linked_shadeops_opt_bc} -o ${linked_shadeops_ptx}
        # This script converts all of the .weak functions defined in the PTX into .visible functions.
        COMMAND ${Python_EXECUTABLE} "${CMAKE_SOURCE_DIR}/src/build-scripts/process-ptx.py"
            ${linked_shadeops_ptx} ${linked_shadeops_ptx}
            DEPENDS ${shadeops_bc_list} ${exec_headers} ${PROJECT_PUBLIC_HEADERS} ${shadeops_srcs} ${headers}
                "${CMAKE_SOURCE_DIR}/src/build-scripts/process-ptx.py"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

    # Serialize the linked bitcode into a CPP file that can be embedded
    # in the current target binary
    add_custom_command ( OUTPUT ${shadeops_bc_cuda_cpp}
        COMMAND ${Python_EXECUTABLE} "${CMAKE_SOURCE_DIR}/src/build-scripts/serialize-bc.py"
            ${linked_shadeops_opt_bc} ${shadeops_bc_cuda_cpp} "rend_llvm_compiled_ops"

        DEPENDS "${CMAKE_SOURCE_DIR}/src/build-scripts/serialize-bc.py" ${linked_shadeops_opt_bc}
            ${exec_headers} ${PROJECT_PUBLIC_HEADERS}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

    install (FILES ${linked_shadeops_ptx}
        DESTINATION ${OSL_PTX_INSTALL_DIR})
endmacro ()
