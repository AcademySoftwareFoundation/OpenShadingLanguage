SET ( liboslnoise_srcs gabornoise.cpp simplexnoise.cpp )

#FILE ( GLOB compiler_headers "../liboslexec/*.h" )
INCLUDE_DIRECTORIES ( ../liboslexec )

if (BUILDSTATIC)
    ADD_LIBRARY ( oslnoise STATIC ${liboslnoise_srcs} )
else ()
    ADD_LIBRARY ( oslnoise SHARED ${liboslnoise_srcs} )
endif ()

TARGET_LINK_LIBRARIES ( oslnoise ${OPENIMAGEIO_LIBRARIES} ${ILMBASE_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} )

install_targets (oslnoise)


if (OSL_BUILD_TESTS)
    add_executable (oslnoise_test oslnoise_test.cpp)
    set_target_properties (oslnoise_test PROPERTIES FOLDER "Unit Tests")
    target_link_libraries (oslnoise_test oslnoise)
    add_test (unit_oslnoise oslnoise_test)
endif()
