# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

# The 'testshade' executable
set ( testshade_srcs testshade.cpp simplerend.cpp optixgridrender.cpp
                     ../testrender/optix_stringtable.cpp )

if (USE_OPTIX)
    set ( testshade_cuda_srcs
        cuda/optix_grid_renderer.cu
        ../testrender/cuda/wrapper.cu )
    set ( testshade_cuda_headers
        ../testrender/cuda/rend_lib.h )

    LLVM_COMPILE_CUDA (
        ${CMAKE_CURRENT_SOURCE_DIR}/../testrender/cuda/rend_lib.cu
        ${testshade_cuda_headers}
        "rend_llvm_compiled_ops"
        rend_lib_bc_cpp
        "-I../testrender/cuda" )

    list (APPEND testshade_srcs ${rend_lib_bc_cpp})

    # Generate PTX for all of the CUDA files
    foreach (cudasrc ${testshade_cuda_srcs})
        NVCC_COMPILE ( ${cudasrc} ptx_generated "-I../testrender/cuda" )
        list (APPEND ptx_list ${ptx_generated})
    endforeach ()

    add_custom_target (testshade_ptx ALL
        DEPENDS ${ptx_list}
        SOURCES ${testshade_cuda_srcs} )

    # Install the PTX files in a fixed location so that they can be
    # loaded at run time
    install (FILES ${ptx_list}
             DESTINATION ${OSL_PTX_INSTALL_DIR})
endif()

add_definitions ("-DPTX_PATH=\"${OSL_PTX_INSTALL_DIR}\"")

add_executable ( testshade ${testshade_srcs} testshademain.cpp )
target_include_directories (testshade
                            BEFORE PRIVATE ${CUDA_INCLUDES} ${OPTIX_INCLUDES})
target_link_libraries (testshade
                       PRIVATE
                           oslexec oslquery
                           ${CUDA_LIBRARIES} ${CUDA_EXTRA_LIBS}  ${OPTIX_LIBRARIES} ${OPTIX_EXTRA_LIBS})
install (TARGETS testshade RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )

# The 'libtestshade' library
add_library ( "libtestshade" ${testshade_srcs} )

set_target_properties (libtestshade
                       PROPERTIES
                       VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
                       SOVERSION ${SOVERSION}
                       OUTPUT_NAME libtestshade${OSL_LIBNAME_SUFFIX}
                       )

target_link_libraries (libtestshade
                       PRIVATE
                           oslexec oslquery
                           ${CUDA_LIBRARIES} ${OPTIX_LIBRARIES} ${OPTIX_EXTRA_LIBS})
set_target_properties (libtestshade PROPERTIES PREFIX "")

install_targets ( libtestshade )

# The 'testshade_dso' executable
add_executable ( testshade_dso testshade_dso.cpp )
target_link_libraries (testshade_dso
                       PRIVATE
                           OpenImageIO::OpenImageIO
                           ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} )
install (TARGETS testshade_dso RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )

