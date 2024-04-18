# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

set ($OSL_EXTRA_NVCC_ARGS "" CACHE STRING "Custom args passed to nvcc when compiling CUDA code")
set (CUDA_OPT_FLAG_NVCC "-O3" CACHE STRING "The optimization level to use when compiling CUDA/C++ files with nvcc")
set (CUDA_OPT_FLAG_CLANG "-O3" CACHE STRING "The optimization level to use when compiling CUDA/C++ files with clang")

# FTZ is enabled by default since it offers the best performance,
# but it may be disabled to debug numerical issues, to better match
# host behavior, etc.
set (CUDA_NO_FTZ OFF CACHE BOOL "Do not enable force-to-zero when compiling for CUDA")
if (CUDA_NO_FTZ)
    add_definitions ("-DOSL_CUDA_NO_FTZ=1")
endif ()

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

    if (NOT CUDA_NO_FTZ)
        set (NVCC_FTZ_FLAG "--ftz=true")
    endif ()

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
            --std=c++${CMAKE_CXX_STANDARD}
            -dc --use_fast_math ${CUDA_OPT_FLAG_NVCC} ${NVCC_FTZ_FLAG}
            --expt-relaxed-constexpr
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
    message (VERBOSE "Current #defines are ${CURRENT_DEFINITIONS}")
    foreach (def ${CURRENT_DEFINITIONS})
        set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} "-D${def}")
    endforeach()
    set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} ${CSTD_FLAGS})

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
    message (VERBOSE "Using LLVM_BC_GENERATOR ${LLVM_BC_GENERATOR} to generate bitcode.")

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

    if (NOT LLVM_OPAQUE_POINTERS AND ${LLVM_VERSION} VERSION_GREATER_EQUAL 15.0)
        # Until we fully support opaque pointers, we need to disable
        # them when using LLVM 15.
        list (APPEND LLVM_COMPILE_FLAGS -Xclang -no-opaque-pointers)
    endif ()

    if (NOT CUDA_NO_FTZ)
        set (CLANG_FTZ_FLAG "-fcuda-flush-denormals-to-zero")
    endif ()

    if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.0")
        # The textureReference API was removed in CUDA 12.0, but it's still referenced
        # in the clang headers, so compilation will fail with 12.0 and newer toolkits
        # due to the missing definitions.
        #
        # We don't actually require any of the clang CUDA texture intrinsics, so we can
        # side-step the issue by preventing them from being included by the preprocessor.
        set (CUDA_TEXREF_FIX "-D__CLANG_CUDA_TEXTURE_INTRINSICS_H__")
    endif()

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
            ${LLVM_COMPILE_FLAGS} ${CUDA_LIB_FLAGS} ${CLANG_MSVC_FIX} ${CUDA_TEXREF_FIX}
            -D__CUDACC__ -DOSL_COMPILING_TO_BITCODE=1 -DNDEBUG -DOIIO_NO_SSE -D__CUDADEVRT_INTERNAL__
            --language=cuda --cuda-device-only --cuda-gpu-arch=${CUDA_TARGET_ARCH}
            -Wno-deprecated-register -Wno-format-security
            -fno-math-errno -ffast-math ${CUDA_OPT_FLAG_CLANG} ${CLANG_FTZ_FLAG} -S -emit-llvm ${extra_clang_args}
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

function ( MAKE_EMBEDDED_CPP symbol_name output_cpp input_file )
    add_custom_command ( OUTPUT ${output_cpp}
        COMMAND ${Python_EXECUTABLE} "${CMAKE_SOURCE_DIR}/src/build-scripts/serialize-bc.py"
            ${input_file} ${output_cpp} "${symbol_name}"

        DEPENDS "${CMAKE_SOURCE_DIR}/src/build-scripts/serialize-bc.py" ${linked_bc}
            ${input_file} ${exec_headers} ${PROJECT_PUBLIC_HEADERS}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
endfunction ()

function ( CUDA_SHADEOPS_COMPILE prefix output_bc output_ptx input_srcs headers )
    set (linked_bc "${CMAKE_CURRENT_BINARY_DIR}/linked_${prefix}.bc")
    set (linked_ptx "${CMAKE_CURRENT_BINARY_DIR}/${prefix}.ptx")
    set (${output_bc} ${linked_bc} PARENT_SCOPE )
    set (${output_ptx} ${linked_ptx} PARENT_SCOPE )

    foreach ( shadeops_src ${input_srcs} )
        MAKE_CUDA_BITCODE ( ${shadeops_src} "_cuda" shadeops_bc "" )
        list ( APPEND shadeops_bc_list ${shadeops_bc} )
    endforeach ()

    if (LLVM_NEW_PASS_MANAGER)
      # There is no --nvptx-assign-valid-global-names flag for the new
      # pass manager, but it appears to run this pass by default.
      string(REPLACE "-O" "O" opt_tool_flags ${CUDA_OPT_FLAG_CLANG})
      set (opt_tool_flags -passes="default<${opt_tool_flags}>")
    else()
      set (opt_tool_flags ${CUDA_OPT_FLAG_CLANG} --nvptx-assign-valid-global-names)
    endif ()

    # Link all of the individual LLVM bitcode files, and emit PTX for the linked bitcode
    add_custom_command ( OUTPUT ${linked_bc} ${linked_ptx}
        COMMAND ${LLVM_LINK_TOOL} ${shadeops_bc_list} -o ${linked_bc}
        COMMAND ${LLVM_OPT_TOOL} ${opt_tool_flags} ${linked_bc} -o ${linked_bc}
        COMMAND ${LLVM_LLC_TOOL} --march=nvptx64 -mcpu=${CUDA_TARGET_ARCH} ${linked_bc} -o ${linked_ptx}
        # This script converts all of the .weak functions defined in the PTX into .visible functions.
        COMMAND ${Python_EXECUTABLE} "${CMAKE_SOURCE_DIR}/src/build-scripts/process-ptx.py"
            ${linked_ptx} ${linked_ptx}
            DEPENDS ${shadeops_bc_list} ${exec_headers} ${PROJECT_PUBLIC_HEADERS} ${input_srcs} ${headers}
                "${CMAKE_SOURCE_DIR}/src/build-scripts/process-ptx.py"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
endfunction ()
