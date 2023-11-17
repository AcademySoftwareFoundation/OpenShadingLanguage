# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

set (local_lib oslnoise)
set (liboslnoise_srcs gabornoise.cpp simplexnoise.cpp)

#file ( GLOB compiler_headers "../liboslexec/*.h" )

add_library (${local_lib} ${liboslnoise_srcs})
target_include_directories (${local_lib}
    PUBLIC
        ${CMAKE_INSTALL_FULL_INCLUDEDIR}
        ${IMATH_INCLUDES}
    PRIVATE
        ../liboslexec)
target_link_libraries (${local_lib}
    PUBLIC
        OpenImageIO::OpenImageIO
        # For OpenEXR/Imath 3.x:
        $<TARGET_NAME_IF_EXISTS:Imath::Imath>
        $<TARGET_NAME_IF_EXISTS:Imath::Half>
        # For OpenEXR >= 2.4/2.5 with reliable exported targets
        $<TARGET_NAME_IF_EXISTS:IlmBase::Imath>
        $<TARGET_NAME_IF_EXISTS:IlmBase::Half>
        $<TARGET_NAME_IF_EXISTS:IlmBase::IlmThread>
        $<TARGET_NAME_IF_EXISTS:IlmBase::Iex>
        # For OpenEXR <= 2.3:
        ${ILMBASE_LIBRARIES}
    PRIVATE
        ${Boost_LIBRARIES} ${CMAKE_DL_LIBS}
    )

target_compile_features (${local_lib}
                         INTERFACE cxx_std_${DOWNSTREAM_CXX_STANDARD})

set (link_flags "${VISIBILITY_MAP_COMMAND} ${EXTRA_DSO_LINK_ARGS}")
if (UNIX AND NOT APPLE)
    # Hide symbols from any static dependent libraries embedded here.
    set (link_flags "${link_flags} -Wl,--exclude-libs,ALL")
endif ()

set_target_properties (${local_lib}
                       PROPERTIES
                       VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
                       SOVERSION ${SOVERSION}
                       OUTPUT_NAME ${local_lib}${OSL_LIBNAME_SUFFIX}
                       LINK_FLAGS  ${link_flags}
                       )

install_targets (${local_lib})


if (OSL_BUILD_TESTS AND BUILD_TESTING)
    add_executable (oslnoise_test oslnoise_test.cpp)
    set_target_properties (oslnoise_test PROPERTIES FOLDER "Unit Tests")
    target_link_libraries (oslnoise_test PRIVATE oslnoise)
    add_test (unit_oslnoise ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/oslnoise_test)
endif()
