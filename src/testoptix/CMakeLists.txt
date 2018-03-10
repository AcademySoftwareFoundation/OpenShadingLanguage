# The 'textoptix' executable

# Modified macro from liboslexec to compile the renderer support library
# from CUDA to LLVM bitcode, and then embed the bitcode in a C++ file to
# be compiled into the testoptix executable.
MACRO ( LLVM_COMPILE llvm_src srclist )
    GET_FILENAME_COMPONENT ( llvmsrc_we ${llvm_src} NAME_WE )
    SET ( llvm_asm "${CMAKE_CURRENT_BINARY_DIR}/${llvmsrc_we}.s" )
    SET ( llvm_bc "${CMAKE_CURRENT_BINARY_DIR}/${llvmsrc_we}.bc" )
    SET ( llvm_bc_cpp "${CMAKE_CURRENT_BINARY_DIR}/${llvmsrc_we}.bc.cpp" )
    if (VERBOSE)
        MESSAGE (STATUS "LLVM_COMPILE (rend_lib) in=${llvm_src}")
        MESSAGE (STATUS "LLVM_COMPILE (rend_lib) bc=${llvm_bc}")
        MESSAGE (STATUS "LLVM_COMPILE (rend_lib) cpp=${llvm_bc_cpp}")
    endif ()
    SET ( ${srclist} ${${srclist}} ${llvm_bc_cpp} )
    get_property (CURRENT_DEFINITIONS DIRECTORY PROPERTY COMPILE_DEFINITIONS)
    if (VERBOSE)
        message (STATUS "Current #defines are ${CURRENT_DEFINITIONS}")
    endif ()
    foreach (def ${CURRENT_DEFINITIONS})
        set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} "-D${def}")
    endforeach()
    set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} ${SIMD_COMPILE_FLAGS} ${CSTD_FLAGS})

    # Figure out what program we will use to make the bitcode.
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
        message (STATUS "Using ${LLVM_BC_GENERATOR} to generate bitcode.")
    endif()

    # Command to turn the renderer library source file into LLVM bitcode .bc,
    # then back into a C++ file with the bc embedded!
    # This macro is set up to accept a .cu file, but could be adapted to accept
    # a generic C++ file.
    ADD_CUSTOM_COMMAND ( OUTPUT ${llvm_bc_cpp}
      COMMAND ${LLVM_BC_GENERATOR}
          "-I${OPTIX_INCLUDE_DIR}"
          "-I${CUDA_INCLUDE_DIR}"
          -D__CUDACC__ -D__CUDA_ARCH__=350
          --cuda-gpu-arch=sm_35 --cuda-device-only --cuda-path="${CUDA_TOOLKIT_ROOT_DIR}"
          -O3 -Wno-format-security -emit-llvm
          ${llvm_src} -c -o ${llvm_bc}
      COMMAND "${CMAKE_SOURCE_DIR}/src/liboslexec/serialize-bc.bash" ${llvm_bc} ${llvm_bc_cpp} "rend_llvm_compiled_ops"
      MAIN_DEPENDENCY ${llvm_src}
      DEPENDS "${CMAKE_SOURCE_DIR}/src/liboslexec/serialize-bc.bash"
              ${llvm_src}
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cuda" )
ENDMACRO ()

# Compile a CUDA file to PTX using NVCC.
MACRO ( NVCC_COMPILE cuda_src ptxlist )
    GET_FILENAME_COMPONENT ( cuda_src_we ${cuda_src} NAME_WE )
    SET ( cuda_ptx "${CMAKE_CURRENT_BINARY_DIR}/${cuda_src_we}.ptx" )
    SET ( ${ptxlist} ${${ptxlist}} ${cuda_ptx} )

    ADD_CUSTOM_COMMAND ( OUTPUT ${cuda_ptx}
        COMMAND ${CUDA_NVCC_EXECUTABLE}
            "-I${OPTIX_INCLUDE_DIR}"
            "-I${CUDA_INCLUDE_DIR}"
            "-I${OPENIMAGEIO_INCLUDE_DIR}"
            "-I${ILMBASE_INCLUDE_DIR}"
            "-I${CMAKE_BINARY_DIR}/include"
            "-I${PROJECT_SOURCE_DIR}/src/include"
            -DOSL_USE_FAST_MATH=1
            -m64 -arch sm_35 -ptx --std=c++11 -O3 --use_fast_math --expt-relaxed-constexpr
            ${cuda_src} -o ${cuda_ptx}
        MAIN_DEPENDENCY ${cuda_src}
        DEPENDS ${cuda_src} oslexec
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
ENDMACRO ()

set ( testoptix_srcs
    shading.cpp
    testoptix.cpp
    )

set ( testoptix_cuda_srcs
    cuda/quad.cu
    cuda/renderer.cu
    cuda/sphere.cu
    cuda/wrapper.cu
    )

LLVM_COMPILE ( ${CMAKE_CURRENT_SOURCE_DIR}/cuda/rend_lib.cu testoptix_srcs )

# Generate PTX for all of the CUDA files
foreach (cudasrc ${testoptix_cuda_srcs})
    NVCC_COMPILE ( ${cudasrc} ptx_list )
endforeach ()

ADD_CUSTOM_TARGET ( testoptix_ptx ALL
    DEPENDS ${ptx_list}
    SOURCES ${testoptix_cuda_srcs} )

ADD_EXECUTABLE ( testoptix ${testoptix_srcs} )
TARGET_LINK_LIBRARIES ( testoptix oslexec oslquery ${CUDA_LIBRARIES} ${OPTIX_LIBRARIES} ${OPTIX_EXTRA_LIBS} ${OPENIMAGEIO_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} )

INSTALL ( TARGETS testoptix RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )

# Install the PTX files in a fixed location so that they can be loaded at run time
INSTALL ( FILES ${ptx_list} DESTINATION ${CMAKE_INSTALL_PREFIX}/ptx/testoptix )
add_definitions ("-DPTX_PATH=\"${CMAKE_INSTALL_PREFIX}/ptx/testoptix\"")
