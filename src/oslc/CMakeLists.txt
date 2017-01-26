SET ( oslc_srcs oslcmain.cpp )

# don't want to link oslexec but oslcomp uses these symbols
if (BUILDSTATIC)
    LIST(APPEND oslc_srcs
        ../liboslexec/oslexec.cpp
        ../liboslexec/typespec.cpp)
endif ()

if (MSVC)
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	# Enable PDB generation for Release build
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
	set (CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG")
endif ()

ADD_EXECUTABLE ( oslc ${oslc_srcs} )
TARGET_LINK_LIBRARIES ( oslc oslcomp ${OPENIMAGEIO_LIBRARY} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
INSTALL ( TARGETS oslc RUNTIME DESTINATION bin )

