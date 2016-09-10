SET ( liboslnoise_srcs gabornoise.cpp simplexnoise.cpp )

#FILE ( GLOB compiler_headers "../liboslexec/*.h" )
INCLUDE_DIRECTORIES ( ../liboslexec )

if (BUILDSTATIC)
    ADD_LIBRARY ( oslnoise STATIC ${liboslnoise_srcs} )
else ()
    ADD_LIBRARY ( oslnoise SHARED ${liboslnoise_srcs} )
endif ()

TARGET_LINK_LIBRARIES ( oslnoise ${OPENIMAGEIO_LIBRARY} ${ILMBASE_LIBRARIES} ${Boost_LIBRARIES} )

INSTALL ( TARGETS oslnoise RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib )


if (OSL_BUILD_TESTS)
    add_executable (oslnoise_test oslnoise_test.cpp)
    set_target_properties (oslnoise_test PROPERTIES FOLDER "Unit Tests")
    target_link_libraries (oslnoise_test oslnoise)
    add_test (unit_oslnoise oslnoise_test)
endif()
