SET ( liboslexec_srcs
          shadingsys.cpp closure.cpp
          bsdf.cpp bsdf_diffuse.cpp bsdf_microfacet.cpp bsdf_phong.cpp
          bsdf_westin.cpp bsdf_ashikhmin_velvet.cpp
          bsdf_cloth_fncs.cpp bsdf_cloth.cpp
          bsdf_reflection.cpp bsdf_refraction.cpp bsdf_transparent.cpp bsdf_ward.cpp
          bssrdf.cpp bsdf_hair.cpp
          emissive.cpp background.cpp
          context.cpp exec.cpp instance.cpp 
          loadshader.cpp master.cpp 
          oparray.cpp opassign.cpp opattribute.cpp
          opcolor.cpp opcompare.cpp opcontrol.cpp opderivs.cpp opinteger.cpp
          opmath.cpp opmathfunc.cpp opmatrix.cpp opmisc.cpp opnoise.cpp 
          opstring.cpp optexture.cpp opvector.cpp
          oslexec.cpp osoreader.cpp
          rendservices.cpp typespec.cpp
    )
FILE ( GLOB compiler_headers "*.h" )

FLEX_BISON ( osolex.l osogram.y oso liboslexec_srcs compiler_headers )

ADD_LIBRARY ( oslexec SHARED ${liboslexec_srcs} )
TARGET_LINK_LIBRARIES ( oslexec oslcomp ${OPENIMAGEIO_LIBRARY} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
LINK_ILMBASE ( oslexec )

INSTALL ( TARGETS oslexec LIBRARY DESTINATION lib )



# Unit tests
add_executable (closure_test closure_test.cpp)
target_link_libraries ( closure_test oslexec ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
link_ilmbase (closure_test)
add_test (unit_closure ${CMAKE_BINARY_DIR}/liboslexec/closure_test)
