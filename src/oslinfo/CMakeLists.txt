SET ( oslinfo_srcs oslinfo.cpp )

if (MSVC)
	# Enable PDB generation for Release build
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi /DEBUG")
endif ()

ADD_EXECUTABLE ( oslinfo ${oslinfo_srcs} )
TARGET_LINK_LIBRARIES ( oslinfo oslquery ${OPENIMAGEIO_LIBRARY} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
INSTALL ( TARGETS oslinfo RUNTIME DESTINATION bin )
