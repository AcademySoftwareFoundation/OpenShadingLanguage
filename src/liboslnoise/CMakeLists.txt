# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

set (liboslnoise_srcs gabornoise.cpp simplexnoise.cpp)

#file ( GLOB compiler_headers "../liboslexec/*.h" )

add_library (oslnoise ${liboslnoise_srcs})
target_include_directories (oslnoise
    PUBLIC
        ${CMAKE_INSTALL_FULL_INCLUDEDIR}
        ${OPENIMAGEIO_INCLUDES}
        ${ILMBASE_INCLUDES}
        ${IMATH_INCLUDES}
    PRIVATE
        ../liboslexec)
target_link_libraries (oslnoise
    PUBLIC
        ${OPENIMAGEIO_LIBRARIES}
        # For OpenEXR/Imath 3.x:
        $<$<TARGET_EXISTS:Imath::Imath>:Imath::Imath>
        $<$<TARGET_EXISTS:Imath::Half>:Imath::Half>
        # For OpenEXR >= 2.4/2.5 with reliable exported targets
        $<$<TARGET_EXISTS:IlmBase::Imath>:IlmBase::Imath>
        $<$<TARGET_EXISTS:IlmBase::Half>:IlmBase::Half>
        $<$<TARGET_EXISTS:IlmBase::IlmThread>:IlmBase::IlmThread>
        $<$<TARGET_EXISTS:IlmBase::Iex>:IlmBase::Iex>
        # For OpenEXR <= 2.3:
        ${ILMBASE_LIBRARIES}
    PRIVATE
        ${Boost_LIBRARIES} ${CMAKE_DL_LIBS}
    )

set_target_properties (oslnoise
                       PROPERTIES
                       VERSION ${OSL_VERSION_MAJOR}.${OSL_VERSION_MINOR}.${OSL_VERSION_PATCH}
                       SOVERSION ${SOVERSION}
                       OUTPUT_NAME oslnoise${OSL_LIBNAME_SUFFIX}
                       )

install_targets (oslnoise)


if (OSL_BUILD_TESTS)
    add_executable (oslnoise_test oslnoise_test.cpp)
    set_target_properties (oslnoise_test PROPERTIES FOLDER "Unit Tests")
    target_link_libraries (oslnoise_test PRIVATE oslnoise)
    add_test (unit_oslnoise ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/oslnoise_test)
endif()
