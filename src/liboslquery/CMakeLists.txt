SET ( liboslquery_srcs oslquery.cpp querystub.cpp 
      ../liboslexec/typespec.cpp )

FILE ( GLOB compiler_headers "../liboslexec/*.h" )
INCLUDE_DIRECTORIES ( ../liboslexec )

FLEX_BISON ( ../liboslexec/osolex.l ../liboslexec/osogram.y oso liboslquery_srcs compiler_headers )

if (MSVC)
	# Enable PDB generation for Release build
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi /DEBUG")
endif ()

if (BUILDSTATIC)
    ADD_LIBRARY ( oslquery STATIC ${liboslquery_srcs} )
else ()
    ADD_LIBRARY ( oslquery SHARED ${liboslquery_srcs} )
endif ()

TARGET_LINK_LIBRARIES ( oslquery ${OPENIMAGEIO_LIBRARY} ${ILMBASE_LIBRARIES} ${Boost_LIBRARIES} )

INSTALL ( TARGETS oslquery RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib )
