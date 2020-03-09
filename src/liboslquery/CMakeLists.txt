set (local_lib oslquery)
file (GLOB lib_src "*.cpp" ../liboslexec/typespec.cpp)
file (GLOB compiler_headers "../liboslexec/*.h")

FLEX_BISON (../liboslexec/osolex.l ../liboslexec/osogram.y oso lib_src compiler_headers)

add_library (${local_lib} ${lib_src})
target_include_directories (${local_lib}
    PUBLIC
        ${CMAKE_INSTALL_FULL_INCLUDEDIR}
    PRIVATE
        ../liboslexec)
target_compile_definitions (${local_lib} PRIVATE OSL_EXPORTS)
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

if (VISIBILITY_COMMAND)
    set_property (TARGET ${local_lib}
                  APPEND PROPERTY LINK_FLAGS ${VISIBILITY_MAP_COMMAND})
endif ()

install_targets (${local_lib})
