SET ( liboslexec_srcs context.cpp exec.cpp instance.cpp 
          loadshader.cpp master.cpp 
          opassign.cpp opcolor.cpp opcompare.cpp opcontrol.cpp opinteger.cpp 
          opmath.cpp opmathfunc.cpp opmatrix.cpp opmisc.cpp 
          opstring.cpp opvector.cpp
          oslexec.cpp osoreader.cpp shadingsys.cpp 
    )
FILE ( GLOB compiler_headers "*.h" )

FLEX_BISON ( osolex.l osogram.y oso liboslexec_srcs compiler_headers )

ADD_LIBRARY ( oslexec SHARED ${liboslexec_srcs} )
TARGET_LINK_LIBRARIES ( oslexec oslcomp ${OPENIMAGEIO_LIBRARY} ${Boost_LIBRARIES} )
LINK_ILMBASE ( oslexec )

INSTALL ( TARGETS oslexec LIBRARY DESTINATION lib )

