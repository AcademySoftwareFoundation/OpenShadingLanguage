SET ( oslc_srcs oslcmain.cpp )

# don't want to link oslexec but oslcomp uses these symbols
if (BUILDSTATIC)
    LIST(APPEND oslc_srcs
        ../liboslexec/oslexec.cpp
        ../liboslexec/typespec.cpp)
endif ()

ADD_EXECUTABLE ( oslc ${oslc_srcs} )
TARGET_LINK_LIBRARIES ( oslc oslcomp ${OPENIMAGEIO_LIBRARY} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
INSTALL ( TARGETS oslc RUNTIME DESTINATION bin )

