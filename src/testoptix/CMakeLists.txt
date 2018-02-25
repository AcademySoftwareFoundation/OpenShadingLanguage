# The 'textoptix' executable

# Compile a CUDA file to PTX using NVCC.
MACRO ( NVCC_COMPILE cuda_src ptxlist )
    GET_FILENAME_COMPONENT ( cuda_src_we ${cuda_src} NAME_WE )
    SET ( cuda_ptx "${CMAKE_CURRENT_BINARY_DIR}/${cuda_src_we}.cu.ptx" )
    SET ( ${ptxlist} ${${ptxlist}} ${cuda_ptx} )

    ADD_CUSTOM_COMMAND ( OUTPUT ${cuda_ptx}
        COMMAND ${CUDA_NVCC_EXECUTABLE}
            -m64 -arch sm_35 -ptx --std=c++11 -O3 --use_fast_math
            "-I${CUDA_INCLUDE_DIR}"
            "-I${OPTIX_INCLUDE_DIR}"
            "-I${CMAKE_SOURCE_DIR}/cuda"
            ${cuda_src} -o ${cuda_ptx}
        MAIN_DEPENDENCY ${cuda_src}
        DEPENDS ${cuda_src} oslexec
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
ENDMACRO ()

set ( testoptix_srcs
    testoptix.cpp
    )

set ( testoptix_cuda_srcs
    cuda/quad.cu
    cuda/renderer.cu
    cuda/sphere.cu
    )

ADD_EXECUTABLE ( testoptix ${testoptix_srcs} )
TARGET_LINK_LIBRARIES ( testoptix oslexec oslquery ${CUDA_LIBRARIES} ${OPTIX_LIBRARIES} ${OPENIMAGEIO_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} lzma )

INSTALL ( TARGETS testoptix RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )

# Generate PTX for all of the CUDA files
foreach (cudasrc ${testoptix_cuda_srcs})
    NVCC_COMPILE ( ${cudasrc} ptx_list )
endforeach ()

ADD_CUSTOM_TARGET ( testoptix_ptx ALL
    DEPENDS ${ptx_list}
    SOURCES ${testoptix_cuda_srcs} )

# Install the PTX files in a fixed location so that they can be loaded at run time
INSTALL ( FILES ${ptx_list} DESTINATION ${CMAKE_INSTALL_PREFIX}/ptx/testoptix )
add_definitions ("-DPTX_PATH=\"${CMAKE_INSTALL_PREFIX}/ptx/testoptix\"")
