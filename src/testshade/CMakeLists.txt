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

    LIST (APPEND testshade_srcs ${rend_lib_bc_cpp})

    # Generate PTX for all of the CUDA files
    foreach (cudasrc ${testshade_cuda_srcs})
        NVCC_COMPILE ( ${cudasrc} ptx_generated "-I../testrender/cuda" )
        LIST (APPEND ptx_list ${ptx_generated})
    endforeach ()

    ADD_CUSTOM_TARGET ( testshade_ptx ALL
        DEPENDS ${ptx_list}
        SOURCES ${testshade_cuda_srcs} )

    # Install the PTX files in a fixed location so that they can be
    # loaded at run time
    INSTALL (FILES ${ptx_list}
             DESTINATION ${OSL_PTX_INSTALL_DIR})
endif()

add_definitions ("-DPTX_PATH=\"${OSL_PTX_INSTALL_DIR}\"")

ADD_EXECUTABLE ( testshade ${testshade_srcs} testshademain.cpp )
TARGET_LINK_LIBRARIES (testshade oslexec oslquery oslcomp
                       ${OPENIMAGEIO_LIBRARIES} ${OCIO_LIBRARIES}
                       ${OPENEXR_LIBRARIES} ${CUDA_LIBRARIES}
                       ${OPTIX_LIBRARIES} ${OPTIX_EXTRA_LIBS}
                       ${Boost_LIBRARIES}
                       ${TIFF_LIBRARIES} ${JPEG_LIBRARIES} ${PNG_LIBRARIES}
                       ${CMAKE_DL_LIBS})
INSTALL ( TARGETS testshade RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )

# The 'libtestshade' library
if (BUILDSTATIC)
    ADD_LIBRARY ( "libtestshade" STATIC ${testshade_srcs} )
else ()
    ADD_LIBRARY ( "libtestshade" SHARED ${testshade_srcs} )
endif ()

set_target_properties (libtestshade
                       PROPERTIES
                       VERSION ${OSL_VERSION_MAJOR}.${OSL_VERSION_MINOR}.${OSL_VERSION_PATCH}
                       SOVERSION ${SOVERSION}
                       OUTPUT_NAME libtestshade${OSL_LIBNAME_SUFFIX}
                       )

TARGET_LINK_LIBRARIES (libtestshade oslexec oslquery
                       ${OPENIMAGEIO_LIBRARIES} ${OPENEXR_LIBRARIES}
                       ${CUDA_LIBRARIES} ${OPTIX_LIBRARIES} ${OPTIX_EXTRA_LIBS}
                       ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} )
SET_TARGET_PROPERTIES (libtestshade PROPERTIES PREFIX "")

install_targets ( libtestshade )

# The 'testshade_dso' executable
ADD_EXECUTABLE ( testshade_dso testshade_dso.cpp )
TARGET_LINK_LIBRARIES ( testshade_dso
                       ${OPENIMAGEIO_LIBRARIES} ${OPENEXR_LIBRARIES}
                       ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} )
INSTALL ( TARGETS testshade_dso RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )

