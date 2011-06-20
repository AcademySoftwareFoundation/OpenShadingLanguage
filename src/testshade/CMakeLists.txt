# The 'testshade' executable
SET ( testshade_srcs testshade.cpp simplerend.cpp )
ADD_EXECUTABLE ( testshade ${testshade_srcs} testshademain.cpp )
LINK_ILMBASE ( testshade )
TARGET_LINK_LIBRARIES ( testshade oslexec oslcomp oslquery ${OPENIMAGEIO_LIBRARY} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
INSTALL ( TARGETS testshade RUNTIME DESTINATION bin )

# The 'libtestshade' dynamic library
add_library ("libtestshade" SHARED ${testshade_srcs})
target_link_libraries (libtestshade oslexec oslcomp oslquery ${OPENIMAGEIO_LIBRARY} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${LLVM_LIBRARY} )
set_target_properties (libtestshade PROPERTIES PREFIX "")
install (TARGETS libtestshade
#         RUNTIME DESTINATION "${BINDIR}" COMPONENT user
         LIBRARY DESTINATION lib 
#COMPONENT user
#         ARCHIVE DESTINATION "${LIBDIR}" COMPONENT developer
)

# The 'testshade_dso' executable
ADD_EXECUTABLE ( testshade_dso testshade_dso.cpp )
LINK_ILMBASE ( testshade_dso )
TARGET_LINK_LIBRARIES ( testshade_dso ${OPENIMAGEIO_LIBRARY} ${CMAKE_DL_LIBS} )
#oslexec oslcomp oslquery ${OPENIMAGEIO_LIBRARY} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
INSTALL ( TARGETS testshade_dso RUNTIME DESTINATION bin )

