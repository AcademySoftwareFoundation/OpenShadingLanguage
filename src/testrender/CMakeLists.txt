# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

# The 'testrender' executable
file(GLOB testrender_srcs *.cpp)

if (USE_OPTIX)
    set (testrender_cuda_srcs
         cuda/quad.cu
         cuda/optix_raytracer.cu
         cuda/sphere.cu
         cuda/wrapper.cu )
    set (testrender_cuda_headers
         cuda/rend_lib.h )

    LLVM_COMPILE_CUDA (
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/rend_lib.cu
        ${testrender_cuda_headers}
        "rend_llvm_compiled_ops"
        rend_lib_bc_cpp
        "" )

    list (APPEND testrender_srcs ${rend_lib_bc_cpp})

    # Generate PTX for all of the CUDA files
    foreach (cudasrc ${testrender_cuda_srcs})
        NVCC_COMPILE ( ${cudasrc} ptx_generated "" )
        list (APPEND ptx_list ${ptx_generated})
    endforeach ()

    add_custom_target (testrender_ptx ALL
        DEPENDS ${ptx_list}
        SOURCES ${testrender_cuda_srcs} )

    # Install the PTX files in a fixed location so that they can be
    # loaded at run time
    install (FILES ${ptx_list}
             DESTINATION ${OSL_PTX_INSTALL_DIR})
endif()

add_definitions ("-DPTX_PATH=\"${OSL_PTX_INSTALL_DIR}\"")

add_executable (testrender ${testrender_srcs})
target_include_directories (testrender
                            BEFORE PRIVATE ${CUDA_INCLUDES} ${OPTIX_INCLUDES})
target_link_libraries (testrender
    PRIVATE
        oslexec oslquery
        pugixml
        ${CUDA_LIBRARIES} ${CUDA_EXTRA_LIBS} ${OPTIX_LIBRARIES} ${OPTIX_EXTRA_LIBS})
install ( TARGETS testrender RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )
