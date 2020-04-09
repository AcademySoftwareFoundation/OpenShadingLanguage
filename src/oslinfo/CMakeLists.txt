set ( oslinfo_srcs oslinfo.cpp )
add_executable ( oslinfo ${oslinfo_srcs} )
target_link_libraries (oslinfo
                       PRIVATE oslquery ${OPENIMAGEIO_LIBRARIES}
                               ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
install_targets (oslinfo)
