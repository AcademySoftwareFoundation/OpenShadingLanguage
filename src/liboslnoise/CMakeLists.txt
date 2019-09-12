SET ( liboslnoise_srcs gabornoise.cpp simplexnoise.cpp )

#FILE ( GLOB compiler_headers "../liboslexec/*.h" )
INCLUDE_DIRECTORIES ( ../liboslexec )

if (BUILDSTATIC)
    ADD_LIBRARY ( oslnoise STATIC ${liboslnoise_srcs} )
else ()
    ADD_LIBRARY ( oslnoise SHARED ${liboslnoise_srcs} )
endif ()

set_target_properties (oslnoise
                       PROPERTIES
                       VERSION ${OSL_VERSION_MAJOR}.${OSL_VERSION_MINOR}.${OSL_VERSION_PATCH}
                       SOVERSION ${SOVERSION}
                       OUTPUT_NAME oslnoise${OSL_LIBNAME_SUFFIX}
                       )

TARGET_LINK_LIBRARIES ( oslnoise 
                        ${OPENIMAGEIO_LIBRARIES} ${ILMBASE_LIBRARIES}
                        ${OPENEXR_ILMIMF_LIBRARY}
                        ${TIFF_LIBRARIES} ${JPEG_LIBRARIES} ${PNG_LIBRARIES}
                        ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} )

install_targets (oslnoise)


if (OSL_BUILD_TESTS)
    add_executable (oslnoise_test oslnoise_test.cpp)
    set_target_properties (oslnoise_test PROPERTIES FOLDER "Unit Tests")
    target_link_libraries (oslnoise_test oslnoise)
    add_test (unit_oslnoise oslnoise_test)
endif()
