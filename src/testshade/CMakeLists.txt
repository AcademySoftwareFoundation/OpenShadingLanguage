# The 'testshade' executable
SET ( testshade_srcs testshade.cpp simplerend.cpp )
ADD_EXECUTABLE ( testshade ${testshade_srcs} testshademain.cpp )
TARGET_LINK_LIBRARIES ( testshade oslexec oslquery ${OPENIMAGEIO_LIBRARIES} ${OPENEXR_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
INSTALL ( TARGETS testshade RUNTIME DESTINATION bin )

# The 'libtestshade' library
if (BUILDSTATIC)
    ADD_LIBRARY ( "libtestshade" STATIC ${testshade_srcs} )
else ()
    ADD_LIBRARY ( "libtestshade" SHARED ${testshade_srcs} )
endif ()

TARGET_LINK_LIBRARIES (libtestshade oslexec oslquery ${OPENIMAGEIO_LIBRARIES} ${OPENEXR_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} )
SET_TARGET_PROPERTIES (libtestshade PROPERTIES PREFIX "")

INSTALL ( TARGETS libtestshade RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

# The 'testshade_dso' executable
ADD_EXECUTABLE ( testshade_dso testshade_dso.cpp )
TARGET_LINK_LIBRARIES ( testshade_dso ${OPENIMAGEIO_LIBRARIES} ${OPENEXR_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} )
INSTALL ( TARGETS testshade_dso RUNTIME DESTINATION bin )

