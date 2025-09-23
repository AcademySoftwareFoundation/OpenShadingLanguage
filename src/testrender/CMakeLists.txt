# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# The 'testrender' executable
set (testrender_srcs
     shading.cpp
     simpleraytracer.cpp
     scene.cpp
     bvh.cpp
     testrender.cpp)

find_package(Threads REQUIRED)

if (OSL_USE_OPTIX)
    list (APPEND testrender_srcs optixraytracer.cpp)
    set (testrender_cuda_srcs
        cuda/optix_raytracer.cu
        )

    set (testrender_rend_lib_srcs
        cuda/rend_lib.cu
        ../testshade/rs_simplerend.cpp
        )

    # We need to make sure that the PTX files are regenerated whenever these
    # files change.
    set (testrender_cuda_headers
        cuda/rend_lib.h
        background.h
        optics.h
        render_params.h
        raytracer.h
        sampling.h
        shading.h
        shading.cpp
        simpleraytracer.cpp
        )

    # Generate PTX for all of the CUDA files
    foreach (cudasrc ${testrender_cuda_srcs})
        NVCC_COMPILE ( ${cudasrc} "${testrender_cuda_headers}" ptx_generated "" BSDL)
        list (APPEND ptx_list ${ptx_generated})
    endforeach ()

    # Compile the renderer-supplied shadeops (rend_lib) to LLVM bitcode and PTX
    add_compile_definitions (OSL_LLVM_CUDA_BITCODE)
    CUDA_SHADEOPS_COMPILE ( "rend_lib_testrender"
        rend_lib_bc
        rend_lib_ptx
        "${testrender_rend_lib_srcs}"
        "${testrender_cuda_headers}"
    )

    # Serialize the linked bitcode into a CPP file to be embedded in the current target binary
    set (rend_lib_bc_cuda_cpp "${CMAKE_CURRENT_BINARY_DIR}/rend_lib_cuda.bc.cpp")
    MAKE_EMBEDDED_CPP( "rend_lib_llvm_compiled_ops" ${rend_lib_bc_cuda_cpp} ${rend_lib_bc} )
    list (APPEND testrender_srcs ${rend_lib_bc_cuda_cpp})
    list (APPEND ptx_list ${rend_lib_ptx})

    add_custom_target (testrender_ptx ALL
        DEPENDS ${ptx_list} ${testrender_cuda_headers}
        SOURCES ${testrender_cuda_srcs} )

    # Install the PTX files in a fixed location so that they can be
    # loaded at run time
    install (FILES ${ptx_list}
             DESTINATION ${OSL_PTX_INSTALL_DIR})
endif()

if (CMAKE_COMPILER_IS_INTELCLANG)
    # To better match existing test results
    add_compile_options("-fp-model=precise")
    # Better performance is likely by not requiring a precise floating point
    # model, although with slightly different numerical results.
endif ()

add_executable (testrender ${testrender_srcs})

target_include_directories (testrender BEFORE PRIVATE ${OpenImageIO_INCLUDES})

target_link_libraries (testrender
    PRIVATE
        oslexec oslquery oslcomp BSDL
        pugixml::pugixml
        Threads::Threads)

osl_optix_target (testrender)

install ( TARGETS testrender RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )
