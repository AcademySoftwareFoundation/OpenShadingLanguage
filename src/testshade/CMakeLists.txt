# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# The 'testshade' executable
set ( testshade_srcs
      testshade.cpp
      simplerend.cpp )

if (OSL_BUILD_BATCHED)
    list(APPEND testshade_srcs
        batched_simplerend.cpp)
endif()

if (OSL_USE_OPTIX)
    list (APPEND testshade_srcs optixgridrender.cpp)
    set ( testshade_cuda_srcs
        cuda/optix_grid_renderer.cu
        ../testrender/cuda/wrapper.cu
        )

    set (testshade_rend_lib_srcs
        ../testrender/cuda/rend_lib.cu
        )

    set ( testshade_cuda_headers
        ../testrender/cuda/rend_lib.h )

    # We need to make sure that the PTX files are regenerated whenever these
    # headers change.
    set ( extra_cuda_headers
        render_params.h )

    # Generate PTX for all of the CUDA files
    foreach (cudasrc ${testshade_cuda_srcs})
        NVCC_COMPILE ( ${cudasrc} ${extra_cuda_headers} ptx_generated "-I../testrender/cuda" )
        list (APPEND ptx_list ${ptx_generated})
    endforeach ()

    # Compile the renderer-supplied shadeops (rend_lib) to LLVM bitcode and PTX
    add_definitions (-DOSL_LLVM_CUDA_BITCODE)
    CUDA_SHADEOPS_COMPILE ( "rend_lib_testshade"
        rend_lib_bc
        rend_lib_ptx
        "${testshade_rend_lib_srcs}"
        "${testshade_cuda_headers}"
    )

    # Serialize the rend_lib bitcode into a CPP file to be embedded in the current target binary
    set (rend_lib_bc_cuda_cpp "${CMAKE_CURRENT_BINARY_DIR}/rend_lib_cuda.bc.cpp")
    MAKE_EMBEDDED_CPP( "rend_lib_llvm_compiled_ops" ${rend_lib_bc_cuda_cpp} ${rend_lib_bc} )
    list (APPEND testshade_srcs ${rend_lib_bc_cuda_cpp})
    list (APPEND ptx_list ${rend_lib_ptx})

    add_custom_target (testshade_ptx ALL
        DEPENDS ${ptx_list}
        SOURCES ${testshade_cuda_srcs} )

    # Install the PTX files in a fixed location so that they can be
    # loaded at run time
    install (FILES ${ptx_list}
             DESTINATION ${OSL_PTX_INSTALL_DIR})
endif()

set ( rs_srcs
    rs_simplerend.cpp )

set(include_dirs ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND include_dirs ${CMAKE_SOURCE_DIR}/src/include)
list(APPEND include_dirs ${CMAKE_BINARY_DIR}/include)
list(APPEND include_dirs ${IMATH_INCLUDES})
list(APPEND include_dirs ${OPENEXR_INCLUDES})
list(APPEND include_dirs ${OpenImageIO_INCLUDES})
list(APPEND include_dirs ${Boost_INCLUDE_DIRS})

EMBED_LLVM_BITCODE_IN_CPP ( "${rs_srcs}" "_host" "testshade_llvm_compiled_rs" testshade_srcs "-DOSL_HOST_RS_BITCODE=1" "${include_dirs}")

add_executable ( testshade ${testshade_srcs} testshademain.cpp )

target_link_libraries (testshade
                       PRIVATE
                           oslexec oslquery)

install (TARGETS testshade RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )

osl_optix_target(testshade)


# As explained in PR #39, problems were reported with calling OSL from a
# Houdini plugin.  So this section sets up a version of testshade as a DSO, to
# be sure we can link it into an executable properly.
#
# But the fact that we use testshade.cpp twice -- once for regular testshade,
# once for the DSO version -- is throwing off the code coverage test, so at
# least for now, disable the dso version entirely when CODECOV is on.
if (NOT CODECOV)
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
                               oslexec oslquery)
    set_target_properties (libtestshade PROPERTIES PREFIX "")

    install_targets ( libtestshade )

    # The 'testshade_dso' executable
    add_executable ( testshade_dso testshade_dso.cpp )
    target_link_libraries (testshade_dso
                           PRIVATE
                               OpenImageIO::OpenImageIO
                               ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} )
    install (TARGETS testshade_dso RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )
    osl_optix_target(libtestshade)
endif ()
