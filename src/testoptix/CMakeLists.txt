# The 'textoptix' executable

set ( testoptix_src
  testoptix.cpp
  )

set ( testoptix_cuda_src
  cuda/quad.cu
  cuda/renderer.cu
  cuda/sphere.cu
  cuda/util.h
  )

ADD_EXECUTABLE ( testoptix ${testoptix_src} )
TARGET_LINK_LIBRARIES ( testoptix oslexec oslquery ${OPENIMAGEIO_LIBRARIES} ${Boost_LIBRARIES} ${OPTIX_LIBRARIES} ${CUDA_LIBRARIES} ${CMAKE_DL_LIBS} )

INSTALL ( TARGETS testoptix RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )

# Install the CUDA sources in a fixed location so that they can be loaded at run time
INSTALL ( FILES ${testoptix_cuda_src} DESTINATION ${CMAKE_INSTALL_PREFIX}/cuda/testoptix )

# Set up some pre-processor macros so that the program will know where to look
# for the CUDA sources to feed to NVRTC
add_definitions ("-DOPTIX_INCLUDE_DIR=\"${OPTIX_INCLUDE_DIR}\"")
add_definitions ("-DCUDA_INCLUDE_DIR=\"${CUDA_INCLUDE_DIR}\"")
add_definitions ("-DCUDA_SRC_PATH=\"${CMAKE_INSTALL_PREFIX}/cuda/testoptix\"")
