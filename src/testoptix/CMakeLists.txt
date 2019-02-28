# The 'textoptix' executable

set ( testoptix_srcs
    optixrend.cpp
    shading.cpp
    stringtable.cpp
    testoptix.cpp
    )

set ( testoptix_cuda_srcs
    cuda/quad.cu
    cuda/renderer.cu
    cuda/sphere.cu
    cuda/tex_renderer.cu
    cuda/wrapper.cu
    )

set ( testoptix_cuda_headers
    cuda/rend_lib.h
    )

LLVM_COMPILE_CUDA (
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/rend_lib.cu
    ${testoptix_cuda_headers}
    "rend_llvm_compiled_ops"
    rend_lib_bc_cpp
    )

LIST (APPEND testoptix_srcs ${rend_lib_bc_cpp} )

# Generate PTX for all of the CUDA files
foreach (cudasrc ${testoptix_cuda_srcs})
    NVCC_COMPILE ( ${cudasrc} ptx_generated )
    LIST (APPEND ptx_list ${ptx_generated})
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
