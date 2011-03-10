SET ( liboslcomp_srcs ast.cpp codegen.cpp oslcomp.cpp symtab.cpp typecheck.cpp
      ../liboslexec/oslexec.cpp ../liboslexec/typespec.cpp
    )
FILE ( GLOB compiler_headers "*.h" )

FLEX_BISON ( osllex.l oslgram.y osl liboslcomp_srcs compiler_headers )

ADD_LIBRARY ( oslcomp SHARED ${liboslcomp_srcs} )
TARGET_LINK_LIBRARIES ( oslcomp ${OPENIMAGEIO_LIBRARY} ${Boost_LIBRARIES} )
LINK_ILMBASE ( oslcomp )

IF ( USE_BOOST_WAVE )
    ADD_DEFINITIONS ( -DUSE_BOOST_WAVE )
ENDIF ()

INSTALL ( TARGETS oslcomp LIBRARY DESTINATION lib )

