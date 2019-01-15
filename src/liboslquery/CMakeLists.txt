SET ( liboslquery_srcs oslquery.cpp querystub.cpp 
      ../liboslexec/typespec.cpp )

FILE ( GLOB compiler_headers "../liboslexec/*.h" )
INCLUDE_DIRECTORIES ( ../liboslexec )

FLEX_BISON ( ../liboslexec/osolex.l ../liboslexec/osogram.y oso liboslquery_srcs compiler_headers )

if (BUILDSTATIC)
    ADD_LIBRARY ( oslquery STATIC ${liboslquery_srcs} )
else ()
    ADD_LIBRARY ( oslquery SHARED ${liboslquery_srcs} )
endif ()

set_target_properties (oslquery
                       PROPERTIES
                       VERSION ${OSL_VERSION_MAJOR}.${OSL_VERSION_MINOR}.${OSL_VERSION_PATCH}
                       SOVERSION ${SOVERSION}
                       OUTPUT_NAME oslquery${OSL_LIBNAME_SUFFIX}
                       )

TARGET_LINK_LIBRARIES ( oslquery ${OPENIMAGEIO_LIBRARIES} ${ILMBASE_LIBRARIES} ${Boost_LIBRARIES} )

install_targets (oslquery)
