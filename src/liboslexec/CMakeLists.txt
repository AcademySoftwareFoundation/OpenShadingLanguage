SET ( liboslexec_srcs closure.cpp context.cpp exec.cpp instance.cpp 
          loadshader.cpp master.cpp 
          oparray.cpp opassign.cpp 
          opcolor.cpp opcompare.cpp opcontrol.cpp opinteger.cpp 
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



# Unit tests
add_executable (closure_test closure_test.cpp)
target_link_libraries ( closure_test oslexec ${Boost_LIBRARIES} )
link_ilmbase (closure_test)
add_test (unit_closure ${CMAKE_BINARY_DIR}/liboslexec/closure_test)
