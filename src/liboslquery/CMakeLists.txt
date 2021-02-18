# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

set (local_lib oslquery)
set (lib_src oslquery.cpp querystub.cpp ../liboslexec/typespec.cpp)
file (GLOB compiler_headers "../liboslexec/*.h")

FLEX_BISON (../liboslexec/osolex.l ../liboslexec/osogram.y oso lib_src compiler_headers)

add_library (${local_lib} ${lib_src})
target_include_directories (${local_lib}
    PUBLIC
        ${CMAKE_INSTALL_FULL_INCLUDEDIR}
        ${OPENIMAGEIO_INCLUDES}
        ${ILMBASE_INCLUDES}
    PRIVATE
        ../liboslexec)
target_link_libraries (${local_lib}
    PUBLIC
        ${OPENIMAGEIO_LIBRARIES} ${ILMBASE_LIBRARIES}
    PRIVATE
        ${Boost_LIBRARIES}
    )

set_target_properties (${local_lib}
    PROPERTIES
        VERSION     ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
        SOVERSION   ${SOVERSION}
        OUTPUT_NAME oslquery${OSL_LIBNAME_SUFFIX}
        POSITION_INDEPENDENT_CODE ON
    )

if (VISIBILITY_MAP_COMMAND)
    set_property (TARGET ${local_lib}
                  APPEND PROPERTY LINK_FLAGS ${VISIBILITY_MAP_COMMAND})
endif ()

install_targets (${local_lib})


# from pythonutils.cmake
if (USE_PYTHON AND Python_Development_FOUND)
    checked_find_package (pybind11 2.4.2 REQUIRED)

    file (GLOB python_srcs py_*.cpp)
    setup_python_module (TARGET    pyoslquery
                         MODULE    oslquery
                         SOURCES   ${python_srcs}
                         LIBS      oslquery
                         )
endif ()
