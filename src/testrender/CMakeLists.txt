# The 'testrender' executable
FILE(GLOB testrender_srcs *.cpp)

add_definitions ("-DPTX_PATH=\"${CMAKE_INSTALL_PREFIX}/ptx/testoptix\"")

if (USE_OPTIX)
    set ( testrender_cuda_srcs
        cuda/quad.cu
        cuda/renderer.cu
        cuda/sphere.cu
        cuda/tex_renderer.cu
        cuda/wrapper.cu )
    set ( testrender_cuda_headers
        cuda/rend_lib.h )

    LLVM_COMPILE_CUDA (
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/rend_lib.cu
        ${testrender_cuda_headers}
        "rend_llvm_compiled_ops"
        rend_lib_bc_cpp
        )

    LIST (APPEND testrender_srcs ${rend_lib_bc_cpp} )

    # Generate PTX for all of the CUDA files
    foreach (cudasrc ${testrender_cuda_srcs})
        NVCC_COMPILE ( ${cudasrc} ptx_generated )
        LIST (APPEND ptx_list ${ptx_generated})
    endforeach ()

    ADD_CUSTOM_TARGET ( testrender_ptx ALL
        DEPENDS ${ptx_list}
        SOURCES ${testrender_cuda_srcs} )
    INSTALL ( FILES ${ptx_list} DESTINATION ${CMAKE_INSTALL_PREFIX}/ptx/testoptix )
endif()


ADD_EXECUTABLE ( testrender ${testrender_srcs} )
TARGET_LINK_LIBRARIES ( testrender oslexec oslquery
                       ${CUDA_LIBRARIES} ${OPTIX_LIBRARIES} ${OPTIX_EXTRA_LIBS}
                       ${OPENIMAGEIO_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
INSTALL ( TARGETS testrender RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )
