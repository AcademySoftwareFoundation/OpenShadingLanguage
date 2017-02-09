SET ( oslinfo_srcs oslinfo.cpp )
ADD_EXECUTABLE ( oslinfo ${oslinfo_srcs} )
TARGET_LINK_LIBRARIES ( oslinfo oslquery ${OPENIMAGEIO_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
INSTALL ( TARGETS oslinfo RUNTIME DESTINATION bin )
