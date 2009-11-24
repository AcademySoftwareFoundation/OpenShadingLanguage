SET ( liboslquery_srcs oslquery.cpp ../liboslexec/osoreader.cpp
      ../liboslexec/typespec.cpp )
FILE ( GLOB compiler_headers "../liboslexec/*.h" )
INCLUDE_DIRECTORIES ( ../liboslexec )

FLEX_BISON ( ../liboslexec/osolex.l ../liboslexec/osogram.y oso liboslquery_srcs compiler_headers )

ADD_LIBRARY ( oslquery SHARED ${liboslquery_srcs} )
TARGET_LINK_LIBRARIES ( oslquery ${OPENIMAGEIO_LIBRARY} ${Boost_LIBRARIES} )
LINK_ILMBASE ( oslquery )

INSTALL ( TARGETS oslquery LIBRARY DESTINATION lib )

